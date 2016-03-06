 ---implement k means method
dofile 'extradata.lua'
require 'image'
require 'nn'
require 'unsup'

-- -- load data set

filename = 'extradata_raw.t7'  -- raw file is not normalized

inputsize =  3-- kernal size  
numFilters = 128
nrows = 96
ncols = 96
nsamples = 10000

patches = torch.Tensor(nsamples,64,inputsize,inputsize)
std = 15
--data = torch.load(filename).trainData.data
extradata = torch.load('extradata.t7').trainData.data



function normalize()
   for i = 1,patches:size(1) do
     patches[i] = image.rgb2yuv(patches[i])
   end
  for i=1, 3 do 
    local mean = patches[{{},i,{},{}}]:mean()
    local std = patches[{{},i,{},{}}]:std()
    patches[{{},i,{},{}}]:add(-mean)
    patches[{{},i,{},{}}]:div(std)
  end
  neighborhood = image.gaussian1D(7)
  normalization = nn.SpatialContrastiveNormalization(1, neighborhood):float()
  for i = 1,patches:size(1) do
   patches[{ i,{1},{},{} }] = normalization(patches[{ i,{1},{},{} }])
 end
end


-- select 3,3,3 size patch
function selectPatch(dataset)
  local imageok = false
  local nrows=dataset:size(3)
  local ncols=dataset:size(4)
  local nsamples = dataset:size(1)

  while not imageok do
     --image index
     local i = math.ceil(torch.uniform(1e-12,nsamples))
     local im = dataset:select(1,i)
     -- select some patch for original that contains original + pos
     local ri = math.ceil(torch.uniform(1e-12,nrows-inputsize))
     local ci = math.ceil(torch.uniform(1e-12,ncols-inputsize))
     local patch = im:narrow(2,ri,inputsize)
     patch = patch:narrow(3,ci,inputsize)
     local patchstd = math.min(patch[1]:std(),patch[2]:std(),patch[3]:std())

     if patchstd > std then
        return patch
     end
  end

end


-- for i=1, nsamples do
-- 	local temp =  selectPatch(data)
-- 	patches[i] = temp:clone()
-- end

-- normalize()



-- -- reshape patches to kmeans
-- patches = patches:reshape(nsamples,3*inputsize*inputsize)
-- centroid_1,count = unsup.kmeans(patches, numFilters, 5000)
-- torch.save('centroid_1',centroid_1)


-- centroid_1_reshape = centroid_1:clone()  -- this tensor is used to image display
-- centroid_1_reshape = centroid_1:reshape(numFilters,3,3,3)
-- de = image.toDisplayTensor{input=centroid_1_reshape,
--                       padding=2,
--                        nrow=math.floor(math.sqrt(numFilters)),
--                      symmetric=true}
-- image.save('centriod_1.jpg', de)

--   -- live display
-- _win1_ = image.display{image=de, win=_win1_, legend='Encoder filters', zoom=5}


--centroid_1 = torch.load('centroid_1.1')

-- first select 64  centriods
function weightInit()
  local weight = centroid_1  -- has 128 kernals 128, 3*3*3
  -- shuffle weight
  index = torch.randperm(weight:size(1))
  -- select 64 kernals
  weightVector = torch.Tensor(64,3,3,3)
  i=0
  j=1
  while j<128 do
    local std =  weight[index[j]]:std()
    j= j+1
    if std>0.4 then
      i =i+1
      weightVector[i]=weight[index[j]]:reshape(3,3,3)
    end
    if i==64 then
      print('weightInitied')
      return  weightVector
    end
  end
end

-- collectgarbage()
local vgg = nn.Sequential()

-- building block
local function ConvBNReLU(nInputPlane, nOutputPlane)
  vgg:add(nn.SpatialConvolution(nInputPlane, nOutputPlane, 3,3, 1,1, 1,1))
  vgg:add(nn.SpatialBatchNormalization(nOutputPlane,1e-3))
  vgg:add(nn.ReLU(true))
  return vgg
end


local MaxPooling = nn.SpatialMaxPooling
ConvBNReLU(3,64)        -- 96*96
vgg:add(MaxPooling(4,4,4,4):ceil()) -- 24*24

-- init weight
vgg:get(1).weight = weightInit()
vgg:training()

outputs = torch.Tensor(400,25,64,24,24)
-- batch forward
local bs=25
local ix=1
for i=1,extradata:size(1),bs do
  local output = vgg:forward(extradata:narrow(1,i,bs))  ---(25,64,24,24)
  outputs[ix]=output
  ix=ix+1
end

torch.save('outputs.t7',outputs)
-- outputs = torch.load('outputs.t7')
-- -- select patches from output 

-- for i=1, nsamples do
--   local temp =  selectPatch(outputs)
--   patches[i] = temp:clone()
-- end

-- -- kmeans again on output
-- patches = patches:reshape(nsamples,64*inputsize*inputsize)
-- centroid_2,count = unsup.kmeans(patches, numFilters, 3000)
-- torch.save('centroid_2.2',centroid_2) --numFilters,64,3,3















