
-- require 'cunn'
require 'unsup'
require 'image'
require 'optim'
require 'ae_data'
local c = require 'trepl.colorize'

----------------------------------------------------------------------
-- parse command-line options
--
cmd = torch.CmdLine()
cmd:text()
cmd:text()
cmd:text('Options')
-- general options:
cmd:option('-dir', 'outputs', 'subdirectory to save experiments in')
cmd:option('-seed', 1, 'initial random seed')
cmd:option('-threads', 2, 'threads')

-- for all models:
cmd:option('-model', 'conv', 'auto-encoder class: linear | conv')
cmd:option('-inputsize', 96, 'size of each input patch')
cmd:option('-nfiltersin', 3, 'number of input convolutional filters')
cmd:option('-nfiltersout', 64, 'number of output convolutional filters')
cmd:option('-lambda', 0.001, 'sparsity coefficient')
cmd:option('-beta', 1, 'prediction error coefficient')
cmd:option('-eta', 5e-4, 'learning rate')
cmd:option('-batchsize', 16, 'batch size')
cmd:option('-etadecay', 1e-5, 'learning rate decay')
cmd:option('-momentum', 0.1, 'gradient momentum')
cmd:option('-maxiter', 500000, 'max number of updates')

-- use hessian information for training:
cmd:option('-hessian', false, 'compute diagonal hessian coefficients to condition learning rates')
cmd:option('-hessiansamples', 500, 'number of samples to use to estimate hessian')
cmd:option('-hessianinterval', 10000, 'compute diagonal hessian coefs at every this many samples')
cmd:option('-minhessian', 0.02, 'min hessian to avoid extreme speed up')
cmd:option('-maxhessian', 5000, 'max hessian to avoid extreme slow down')

-- for conv models:
cmd:option('-kernelsize', 3, 'size of convolutional kernels')

-- logging:
cmd:option('-datafile', 'extradata.t7', 'Dataset position: small.t7 | extradata.t7')
cmd:option('-statinterval', 1000, 'interval for saving stats and models')
cmd:option('-v', false, 'be verbose')
cmd:option('-display', true, 'display stuff')
cmd:option('-wcar', '', 'additional flag to differentiate this run')
cmd:text()

params = cmd:parse(arg)

rundir = cmd:string(params.model, params, {dir=true})
params.rundir = params.dir .. '/' .. rundir

if paths.dirp(params.rundir) then
   os.execute('rm -r ' .. params.rundir)
end
os.execute('mkdir -p ' .. params.rundir)

cmd:addTime(params.model)
cmd:log(params.rundir .. '/log.txt', params)

torch.manualSeed(params.seed)
torch.setnumthreads(params.threads)



----------------------------------------------------------------------
-- load data
--
filename = paths.basename(params.datafile)
if params.model == 'linear' then
   params.inputsize = params.kernelsize
end

dataset = getdata(filename, params.inputsize) --{dsample,dsample,im}

if params.display then
   displayData(dataset, 50) 
end

nsamples = dataset:size()

print(nsamples) -- how many unlabeled sample are we using 


-- initilize weight with random select patches

function weightInit(weightvector)
   -- extract patches from images
   -- basic var
   local h = weightvector:size()[3]
   local w = weightvector:size()[2]
   local n = weightvector:size()[1]

   local patches = torch.Tensor(n, w, h)

   for i=1, n do
      -- randomely choose a path
      local rd1, rd2  = torch.random(1,nsamples), torch.random(1,params.inputsize-w)
      local im = dataset[rd1]
      patches[i] = im:narrow(1,rd2,w):narrow(2,rd2,h)
   end

   norm = nn.SpatialBatchNormalization(n,nil,nil,false)
   patches = norm:forward(patches)
   -- normalize 
   if display then
      image.display(patches)
   end

   weightvector = patches:copy()
end

local function MSRinit(net)
  local function init(name)
    for k,v in pairs(net:findModules(name)) do
      local n = v.kW*v.kH*v.nOutputPlane
      v.weight:normal(0,math.sqrt(2/n))
      v.bias:zero()
    end
  end
  -- have to do for both backends
  init'nn.SpatialConvolution'
end


----------------------------------------------------------------------
-- create model

if params.model == 'linear' then
   -- params
   inputSize = params.nfiltersin * params.inputsize * params.inputsize
   outputSize = params.nfiltersout 

   -- encoder
   encoder = nn.Sequential()
   encoder:add(nn.Linear(inputSize,outputSize))
   encoder:add(nn.Tanh())
   -- encoder:add(nn.Diag(outputSize))

   -- decoder is L1 solution
   decoder = nn.Sequential()
   decoder:add(nn.Linear(outputSize,inputSize))

   -- PSD autoencoder
   module = unsup.AutoEncoder(encoder, decoder, params.lambda)

   -- verbose
   print('==> constructed linear  auto-encoder')

elseif params.model == 'conv' then

   -- params:
   conntable = nn.tables.full(params.nfiltersin, params.nfiltersout)
   kw, kh = params.kernelsize, params.kernelsize
   iw, ih = params.inputsize, params.inputsize

   -- connection table:
   local decodertable = conntable:clone()
   decodertable[{ {},1 }] = conntable[{ {},2 }]
   decodertable[{ {},2 }] = conntable[{ {},1 }]
   local outputFeatures = conntable[{ {},2 }]:max()

   -- encoder:
   encoder = nn.Sequential()
   encoder:add(nn.SpatialConvolutionMap(conntable, kw, kh, 1, 1))
   encoder:add(nn.ReLU())
   encoder:add(nn.SpatialMaxPooling(4,4,4,4))  --64  // feature map size : 64 *3 * 23 *23
   -- encoder:add(nn.SpatialBatchNormalization(params.nfiltersout,nil,nil,false)) 

   -- decoder is L1 solution:
   decoder = nn.Sequential()
   decoder:add(nn.SpatialMaxUnpooling(encoder:get(3)))  -- 64*3*92*92
   decoder:add(nn.SpatialFullConvolutionMap(decodertable, kw, kh, 1, 1))

   -- PSD autoencoder
   module = unsup.AutoEncoder(encoder, decoder, params.lambda)
   MSRinit(encoder)

   -- weight initilization  -192*5*5
   -- weightInit(module.encoder.modules[1].weight)


   -- convert dataset to convolutional (returns 1xKxK tensors (3D), instead of K*K (1D))
   dataset:conv()

   -- verbose
   print('==> constructed convolutional  auto-encoder')

else
   print('==> unknown model: ' .. params.model)
   os.exit()

end


-- module:cuda()
----------------------------------------------------------------------
-- trainable parameters
-- are we using the hessian?
if params.hessian then
   nn.hessian.enable()
   module:initDiagHessianParameters()
end

-- get all parameters
x,dl_dx,ddl_ddx = module:getParameters()
----------------------------------------------------------------------

-- train model

print('==> training model')

local avTrainingError = torch.FloatTensor(math.ceil(params.maxiter/params.statinterval)):zero()
local err = 0
local iter = 0

for t = 1,params.maxiter,params.batchsize do

   --------------------------------------------------------------------
   -- update diagonal hessian parameters
   --
   if params.hessian and math.fmod(t , params.hessianinterval) == 1 then
      -- some extra vars:
      local hessiansamples = params.hessiansamples
      local minhessian = params.minhessian
      local maxhessian = params.maxhessian
      local ddl_ddx_avg = ddl_ddx:clone(ddl_ddx):zero()
      etas = etas or ddl_ddx:clone()

      print('==> estimating diagonal hessian elements')
      for i = 1,hessiansamples do
         -- next
         local ex = dataset[i]
         local input = ex[1]
         local target = ex[2]
         module:updateOutput(input, target)

         -- gradient
         dl_dx:zero()
         module:updateGradInput(input, target)
         module:accGradParameters(input, target)

         -- hessian
         ddl_ddx:zero()
         module:updateDiagHessianInput(input, target)
         module:accDiagHessianParameters(input, target)

         -- accumulate
         ddl_ddx_avg:add(1/hessiansamples, ddl_ddx)
      end

      -- cap hessian params
      print('==> ddl/ddx : min/max = ' .. ddl_ddx_avg:min() .. '/' .. ddl_ddx_avg:max())
      ddl_ddx_avg[torch.lt(ddl_ddx_avg,minhessian)] = minhessian
      ddl_ddx_avg[torch.gt(ddl_ddx_avg,maxhessian)] = maxhessian
      print('==> corrected ddl/ddx : min/max = ' .. ddl_ddx_avg:min() .. '/' .. ddl_ddx_avg:max())

      -- generate learning rates
      etas:fill(1):cdiv(ddl_ddx_avg)
   end

   --------------------------------------------------------------------
   iter = iter+1
   xlua.progress(iter, params.statinterval)
   --------------------------------------------------------------------
   -- create mini-batch
   --
   local example = dataset[t]
   local inputs = {}
   local targets = {}
   for i = t,t+params.batchsize-1 do
      -- load new sample
      local sample = dataset[i]
      local input = sample[1]:clone()
      local target = sample[2]:clone()
      table.insert(inputs, input)
      table.insert(targets, target)
   end

   --------------------------------------------------------------------
   -- define eval closure
   --
   local feval = function()
      -- reset gradient/f
      local f = 0
      dl_dx:zero()

      -- estimate f and gradients, for minibatch
      for i = 1,#inputs do
         -- f
         f = f + module:updateOutput(inputs[i], targets[i])

         -- gradients
         module:updateGradInput(inputs[i], targets[i])
         module:accGradParameters(inputs[i], targets[i])
      end

      -- normalize
      dl_dx:div(#inputs)
      f = f/#inputs

      -- return f and df/dx
      return f,dl_dx
   end

   --------------------------------------------------------------------
   -- one SGD step
   --
   sgdconf = sgdconf or {learningRate = params.eta,
                         learningRateDecay = params.etadecay,
                         learningRates = etas,
                         momentum = params.momentum}
   _,fs = optim.sgd(feval, x, sgdconf)
   err = err + fs[1]

   -- normalize
   if params.model:find('linear') then 
      module:normalize()
   end

   --------------------------------------------------------------------
   -- compute statistics / report error
   --
   --print('current position:'..math.fmod(t , params.statinterval))
   if math.fmod(iter, params.statinterval) == 0 then
      -- progress

      -- report
      print('==> iteration = ' .. t .. ', average loss = ' .. err/params.statinterval)

      -- get weights
      eweight = module.encoder.modules[1].weight
      if module.decoder.D then
         dweight = module.decoder.D.weight
      else
         dweight = module.decoder.modules[2].weight
      end

      -- reshape weights if linear matrix is used
      if params.model:find('linear') then
         dweight = dweight:transpose(1,2):unfold(2,params.inputsize,params.inputsize)
         eweight = eweight:unfold(2,params.inputsize,params.inputsize):unfold(2,params.inputsize,params.inputsize)
      end


      -- render filters
      dd = image.toDisplayTensor{input=dweight,
                                 padding=2,
                                 nrow=math.floor(math.sqrt(params.nfiltersout)),
                                 symmetric=true}
      de = image.toDisplayTensor{input=eweight,
                                 padding=2,
                                 nrow=math.floor(math.sqrt(params.nfiltersout)),
                                 symmetric=true}

      -- live display
      if params.display then
        -- _win1_ = image.display{image=dd, win=_win1_, legend='Decoder filters', zoom=5}
         _win2_ = image.display{image=de, win=_win2_, legend='Encoder filters', zoom=5}

      end

      -- save stuff
      --image.save(params.rundir .. '/filters_dec_' .. t .. '.jpg', dd)
      image.save(params.rundir .. '/filters_enc_' .. t .. '.jpg', de)
      torch.save(params.rundir .. '/eweight_' .. t .. '.t7', eweight)
      torch.save(params.rundir .. '/model_' .. t .. '.bin', module)

      -- reset counters
      err = 0; iter = 0
   end


end





-- save final weight to weight.t7
torch.save('eweight_1.t7', eweight)




