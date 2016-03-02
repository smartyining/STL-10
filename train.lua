require 'xlua'
require 'optim'
dofile './provider.lua'
dofile 'augmentation.lua'
require 'cunn'
local c = require 'trepl.colorize'

opt = lapp[[
   -s,--save                  (default "logs")      subdirectory to save logs
   -b,--batchSize             (default 64)          batch size
   -r,--learningRate          (default 1)        learning rate
   --learningRateDecay        (default 1e-7)      learning rate decay
   --weightDecay              (default 0.0005)      weightDecay
   -m,--momentum              (default 0.9)         momentum
   --epoch_step               (default 25)          epoch step
   --model                    (default vgg_bn_drop)     model name
   --max_epoch                (default 300)           maximum number of iterations
   --backend                  (default nn)            backend
]]

print(opt)

model = nn.Sequential()
-- data augmentation block
local function augment()
  model:add(nn.BatchHFlip():float())
  model:add(nn.BatchTranslate():float())
  model:add(nn.BatchScale():float())
  model:add(nn.BatchRotate():float())   
  model:add(nn.BatchContrast1():float())
  model:add(nn.BatchVFlip():float())
  return model
end


print(c.blue '==>' ..' configuring model')
augment()
model:add(nn.Copy('torch.FloatTensor','torch.CudaTensor'):cuda())
model:add(dofile('models/'..opt.model..'.lua'):cuda())
model:get(7).updateGradInput = function(input) return end

collectgarbage()

if opt.backend == 'cudnn' then
   require 'cudnn'
   cudnn.convert(model:get(8), cudnn)
end

print(model)


function weightInit()
  dir = 'model_1.bin'
  local module_pre = torch.load(dir)
  local weightVector = module_pre.encoder.modules[1].weight:cuda()
  return  weightVector
end


print(c.blue '==>' ..' loading data')
provider = torch.load './provider.t7'
provider.trainData.data = provider.trainData.data:float()
provider.valData.data = provider.valData.data:float()

confusion = optim.ConfusionMatrix(10)

print('Will save at '..opt.save)
paths.mkdir(opt.save)
valLogger = optim.Logger(paths.concat(opt.save, 'val.log'))
valLogger:setNames{'% mean class accuracy (train set)', '% mean class accuracy (val set)'}
valLogger.showPlot = false

parameters,gradParameters = model:getParameters()


function init(net)
  local function init(name)
    for k,v in pairs(net:findModules(name)) do
      if v.nInputPlane==3 and v.nOutputPlane==64 then
        print('weight reinilized')
        v.weight= weightInit():clone()
      end
    end
  end
  -- have to do for both backends
  init'nn.SpatialConvolution'
end

--init(model)

print(c.blue'==>' ..' setting criterion')
criterion = nn.CrossEntropyCriterion():cuda()


print(c.blue'==>' ..' configuring optimizer')
optimState = {
  learningRate = opt.learningRate,
  weightDecay = opt.weightDecay,
  momentum = opt.momentum,
  learningRateDecay = opt.learningRateDecay,
}

collectgarbage()

function train()
  model:training()

  epoch = epoch or 1

  -- drop learning rate every "epoch_step" epochs
  if epoch % opt.epoch_step == 0 then optimState.learningRate = optimState.learningRate/2 end
  
  print(c.blue '==>'.." online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')

  local targets = torch.CudaTensor(opt.batchSize)
  local indices = torch.randperm(provider.trainData.data:size(1)):long():split(opt.batchSize)
  -- remove last element so that all the batches have equal size
  indices[#indices] = nil

  local tic = torch.tic()
  for t,v in ipairs(indices) do
    xlua.progress(t, #indices)

    local inputs = provider.trainData.data:index(1,v) -- 64 images
    targets:copy(provider.trainData.labels:index(1,v))

    -- add cotrast2  and color mod
    local m1 = nn.Contrast2()
    local m2 = nn.ColorMod()
    for it=1, opt.batchSize do
        local rand = torch.rand(1)
        local rand2 = torch.rand(1)
        if rand>0.5 then
          inputs[it] = m1:forward(inputs[it])
        end
        if rand2>0.5 then
          inputs[it]=m2:forward(inputs[it])
    end

    local feval = function(x)
      if x ~= parameters then parameters:copy(x) end
      gradParameters:zero()
      
      local outputs = model:forward(inputs)
      local f = criterion:forward(outputs, targets)
      local df_do = criterion:backward(outputs, targets)
      model:backward(inputs, df_do)

      confusion:batchAdd(outputs, targets)

      return f,gradParameters
    end
    optim.sgd(feval, parameters, optimState)
  end

  confusion:updateValids()
  print(('Train accuracy: '..c.cyan'%.2f'..' %%\t time: %.2f s'):format(
        confusion.totalValid * 100, torch.toc(tic)))

  train_acc = confusion.totalValid * 100

  confusion:zero()
  epoch = epoch + 1
end

collectgarbage()
function val()
  -- disable flips, dropouts and batch normalization
  model:evaluate()
  print(c.blue '==>'.." valing")
  local bs = 25
  for i=1,provider.valData.data:size(1),bs do
    -- print(provider.valData.labels:narrow(1,i,bs):size()) 
    local outputs = model:forward(provider.valData.data:narrow(1,i,bs))
    -- print(outputs:size()) 
    -- print(provider.valData.labels:narrow(1,i,bs):size()) 
    confusion:batchAdd(outputs, provider.valData.labels:narrow(1,i,bs))
  end

  confusion:updateValids()
  print('val accuracy:', confusion.totalValid * 100)
  
  if valLogger then
    paths.mkdir(opt.save)
    valLogger:add{train_acc, confusion.totalValid * 100}
    valLogger:style{'-','-'}
    valLogger:plot()

    local base64im
    do
      os.execute(('convert -density 200 %s/val.log.eps %s/val.png'):format(opt.save,opt.save))
      os.execute(('openssl base64 -in %s/val.png -out %s/val.base64'):format(opt.save,opt.save))
      local f = io.open(opt.save..'/val.base64')
      if f then base64im = f:read'*all' end
    end

    local file = io.open(opt.save..'/report.html','w')
    file:write(([[
    <!DOCTYPE html>
    <html>
    <body>
    <title>%s - %s</title>
    <img src="data:image/png;base64,%s">
    <h4>optimState:</h4>
    <table>
    ]]):format(opt.save,epoch,base64im))
    for k,v in pairs(optimState) do
      if torch.type(v) == 'number' then
        file:write('<tr><td>'..k..'</td><td>'..v..'</td></tr>\n')
      end
    end
    file:write'</table><pre>\n'
    file:write(tostring(confusion)..'\n')
    file:write(tostring(model)..'\n')
    file:write'</pre></body></html>'
    file:close()
  end

  -- save model every 50 epoch_step
  if epoch % 5 == 0 then
    local filename = paths.concat(opt.save, 'model.net')
    print('==> saving model to '..filename)
    torch.save(filename, model:get(7))
  end

  confusion:zero()
end


for i=1,opt.max_epoch do
  train()
  val()
  --est()
end


