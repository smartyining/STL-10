----------------------------------------------------------------------
-- This script implements load the previous saved model and
-- generate prediction on test set
----------------------------------------------------------------------
require 'torch'
require 'nn'
require 'image'
require 'optim'
require 'csvigo'
dofile 'provider.lua'

trsize = 5000
testsize = 8000
channel = 3
height = 96
width = 96


-- parse STL-10 data from table into Tensor
function parseDataLabel(d, numSamples, numChannels, height, width)
   local t = torch.ByteTensor(numSamples, numChannels, height, width)
   local l = torch.ByteTensor(numSamples)
   local idx = 1
   for i = 1, #d do
      local this_d = d[i]
      for j = 1, #this_d do
    t[idx]:copy(this_d[j])
    l[idx] = i
    idx = idx + 1
      end
   end
   assert(idx == numSamples+1)
   return t, l
end

print '==> loading dataset'

test_path = 'stl-10/test.t7b'

provider = torch.load 'provider.t7'
trainData = provider.trainData
trainData.data = trainData.data:float()
testData = {
     data = torch.Tensor(),
     labels = torch.Tensor(),
     size = function() return testsize end
}

local raw_test = torch.load(test_path)
testData.data, testData.labels = parseDataLabel(raw_test.data,
                                               testsize, channel, height, width)

testData.data = testData.data:float()
testData.labels = testData.labels:float()
-- load previous saved model
print '==> loading trained model'
model_path='yining/logs'
model = torch.load(paths.concat(model_path,'model.net')):float()


f = io.open('predictions.csv', 'w')
f:write("Id,Prediction\n")

function preproc()
   --preprocessing testing data 

   testData.data = testData.data:float()
   trainData.data = trainData.data:float()

   -- preprocess trainSet
  local normalization = nn.SpatialContrastiveNormalization(1, image.gaussian1D(7))
  for i = 1,trainData:size() do
     xlua.progress(i, trainData:size())
     -- rgb -> yuv
     local rgb = trainData.data[i]
     local yuv = image.rgb2yuv(rgb)
     -- normalize y locally:
     yuv[1] = normalization(yuv[{{1}}])
     trainData.data[i] = yuv
  end

-- normalize u globally:
  local mean_u = trainData.data:select(2,2):mean()
  local std_u = trainData.data:select(2,2):std()
  trainData.data:select(2,2):add(-mean_u)
  trainData.data:select(2,2):div(std_u)
  -- normalize v globally:
  local mean_v = trainData.data:select(2,3):mean()
  local std_v = trainData.data:select(2,3):std()
  trainData.data:select(2,3):add(-mean_v)
  trainData.data:select(2,3):div(std_v)

  trainData.mean_u = mean_u
  trainData.std_u = std_u
  trainData.mean_v = mean_v
  trainData.std_v = std_v

  -- preprocess valSet
  for i = 1,testData:size() do
    xlua.progress(i,testData:size())
     -- rgb -> yuv
     local rgb = testData.data[i]
     local yuv = image.rgb2yuv(rgb)
     -- normalize y locally:
     yuv[{1}] = normalization(yuv[{{1}}])
     testData.data[i] = yuv
  end
  -- normalize u globally:
  testData.data:select(2,2):add(-mean_u)
  testData.data:select(2,2):div(std_u)
  -- normalize v globally:
  testData.data:select(2,3):add(-mean_v)
  testData.data:select(2,3):div(std_v)

end


function test()
  -- disable flips, dropouts and batch normalization
  model:evaluate()
  print(c.blue '==>'.." valing")
  local bs = 25
  for i=1,provider.testData.data:size(1),bs do
    -- print(provider.valData.labels:narrow(1,i,bs):size()) 
    local outputs = model:forward(provider.testData.data:narrow(1,i,bs))
    -- print(outputs:size()) 
    -- print(provider.valData.labels:narrow(1,i,bs):size()) 
    confusion:batchAdd(outputs, provider.testData.labels:narrow(1,i,bs))
  end

  confusion:updateValids()
  print('test accuracy:', confusion.totalValid * 100)

  confusion:zero()
end

function predict()
   -- local vars
   local time = sys.clock()

   -- averaged param use?
   if average then
      cachedparams = parameters:clone()
      parameters:copy(average)
   end

   -- set model to evaluate mode (for modules that differ in training and testing, like Dropout)
   model:evaluate()

   -- test over test data
   print('==> testing on test set:')

   for t = 1,testData:size() do
      -- display progress
      xlua.progress(t, testData:size())
	   f:write(t..",")

      -- get new sample
      local input = testData.data[t]
      input = input:double()
      local target = testData.labels[t]

      -- test sample and write to file
      local pred = model:forward(input)
      val,ind=torch.max(pred,1)        
	   f:write(ind[1]%10)
	   f:write("\n")
   end
   
   f.close()

   -- timing
   time = sys.clock() - time
   time = time / testData:size()
   print("\n==> time to test 1 sample = " .. (time*1000) .. 'ms')

   -- averaged param use?
   if average then
      -- restore parameters
      parameters:copy(cachedparams)
   end
end

preproc()
predict()
test()
