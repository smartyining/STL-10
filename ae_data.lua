dofile 'extradata.lua'

local data_verbose = false

function getdata(datafile, inputsize,std)
   local data = torch.load(datafile).trainData.data
   local dataset ={}

   local std = std or 0.2
   local nsamples = data:size(1)
   local nchannels = data:size(2)
   local nrows = data:size(3)
   local ncols = data:size(4)

   function dataset:size()
      return nsamples
   end

   function dataset:selectPatch(nr,nc)
      local imageok = false
      if simdata_verbose then
         print('selectPatch')
      end
      while not imageok do
         --image index
         local i = math.ceil(torch.uniform(1e-12,nsamples))
         local im = data:select(1,i)
         -- select some patch for original that contains original + pos
         local ri = math.ceil(torch.uniform(1e-12,nrows-nr))
         local ci = math.ceil(torch.uniform(1e-12,ncols-nc))
         local patch = im:narrow(2,ri,nr)
         patch = patch:narrow(3,ci,nc)
         local patchstd = patch:std()
         if data_verbose then
            print('Image ' .. i .. ' ri= ' .. ri .. ' ci= ' .. ci .. ' std= ' .. patchstd)
         end
         if patchstd > std then
            if data_verbose then
               print(patch:min(),patch:max())
            end
            return patch,i,im
         end
      end
   end

   local dsample = torch.Tensor(3*inputsize*inputsize)

   function dataset:conv()
      dsample = torch.Tensor(3,inputsize,inputsize)
   end

   setmetatable(dataset, {__index = function(self, index)
                                       local sample,i,im = self:selectPatch(inputsize, inputsize)
                                       dsample:copy(sample)
                                       return {dsample,dsample,im}
                                    end})
   return dataset
end


-- dataset, dataset=createDataset(....)
-- nsamples, how many samples to display from dataset
-- nrow, number of samples per row for displaying samples
-- zoom, zoom at which to draw dataset

function displayData(dataset, nsamples, nrow, zoom)
   require 'image'
   local nsamples = nsamples or 100
   local zoom = zoom or 5
   local nrow = nrow or 10

   cntr = 1
   local ex = {}
   for i=1,nsamples do
      local exx = dataset[i]
      ex[cntr] = exx[1]:clone():reshape(3,math.sqrt(exx[1]:size(1)/3),math.sqrt(exx[1]:size(1)/3))
      cntr = cntr + 1
   end

   return image.display{image=ex, padding=2, symmetric=true, zoom=zoom, nrow=nrow, legend='Training Data'}
end
