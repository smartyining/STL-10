---- define data augmentation module
require 'torch'
require 'image'
require 'xlua'
require 'unsup'

--- horizontal flip
do
  local BatchHFlip,parent = torch.class('nn.BatchHFlip', 'nn.Module')

  function BatchHFlip:__init()
    parent.__init(self)
    self.train = true
  end

  function BatchHFlip:updateOutput(input)
    if self.train then
      local bs = input:size(1)
      local flip_mask = torch.randperm(bs):le(bs/2)
      for i=1,input:size(1) do
        if flip_mask[i] == 1 then image.hflip(input[i], input[i]) end
      end
    end
    self.output:set(input)
    return self.output
  end
end

---vertical flip
do
  local BatchVFlip,parent = torch.class('nn.BatchVFlip', 'nn.Module')

  function BatchVFlip:__init()
    parent.__init(self)
    self.train = true
  end

  function BatchVFlip:updateOutput(input)
    if self.train then
      local bs = input:size(1)
      local flip_mask = torch.randperm(bs):le(bs/2)
      for i=1,input:size(1) do
        if flip_mask[i] == 1 then image.vflip(input[i], input[i]) end
      end
    end
    self.output:set(input)
    return self.output
  end
end


----------------------------------------------------------------------
do
  local BatchTranslate,parent = torch.class('nn.BatchTranslate', 'nn.Module')

  function BatchTranslate:__init()
    parent.__init(self)
    self.train = true
  end


  function BatchTranslate:updateOutput(input)
    if self.train then
      local patchsize = input:size(3)
      local dist = math.floor((torch.rand(1)[1] * 0.4 - 0.2) * patchsize)
      local bs = input:size(1)
      local flip_mask = torch.randperm(bs):le(bs/2)
      for i=1,input:size(1) do
        if flip_mask[i] == 1 then
          image.translate(input[i],input[i],dist,dist) end
      end
    end
    self.output:set(input)
    return self.output
  end
end


--------------------------------------------------------------------
do
  local BatchScale,parent = torch.class('nn.BatchScale', 'nn.Module')

  function BatchScale:__init()
    parent.__init(self)
    self.train = true
  end

  function BatchScale:updateOutput(input)
    if self.train then

      local patchsize = input:size(3)
      local factor = torch.rand(1)[1] * 0.7 + 0.7
      local img_size = math.floor(patchsize * factor)
      local bs = input:size(1)
      local flip_mask = torch.randperm(bs):le(bs/2)
      for i=1,input:size(1) do
        if flip_mask[i] == 1 then input[i]=image.scale(input[i],img_size,img_size) end
      end
    end
     self.output:set(input)
    return self.output
  end
end



---------------------------------------------------------------------
do
  local BatchRotate,parent = torch.class('nn.BatchRotate', 'nn.Module')

  function BatchRotate:__init()
    parent.__init(self)
    self.train = true
  end

  function BatchRotate:updateOutput(input)
    if self.train then
      local patchsize = input:size(3)
   -- random angle
      local rad = torch.rand(1)[1] * math.pi * 2/9  - math.pi/96
      self.output = input:clone()
      local bs = input:size(1)
      local flip_mask = torch.randperm(bs):le(bs/2)
      for i=1,input:size(1) do
        if flip_mask[i] == 1 then image.rotate(self.output[i],input[i],rad) end
      end
    end
    return self.output
  end
end

-----------------------------------------------------
do
  local BatchContrast1,parent = torch.class('nn.BatchContrast1', 'nn.Module')

  function BatchContrast1:__init()
    parent.__init(self)
    self.train = true
  end

  function BatchContrast1:updateOutput(input)
    if self.train then
      local bs = input:size(1)
      local patchsize = input:size(3)
      local flip_mask = torch.randperm(bs):le(bs/2)
      for i=1,input:size(1) do
        if flip_mask[i] == 1 then 
            local factors = torch.rand(1, 3):mul(1.5):add(0.5)
            local unfolded = input[i]:reshape(3, patchsize*patchsize):transpose(1, 2)
            ce, cv = unsup.pcacov(unfolded)
            local proj = unfolded * cv
            proj:cmul(torch.expand(factors, patchsize*patchsize, 3))
            input[i] = (proj * torch.inverse(cv)):transpose(1, 2):reshape(3, patchsize, patchsize)

         end
      end
    end
    self.output:set(input)
    return self.output
  end
end


------------------------------------------

do
  local BatchContrast2,parent = torch.class('nn.BatchContrast2', 'nn.Module')

  function BatchContrast2:__init()
    parent.__init(self)
    self.train = true
  end

function BatchContrast2:updateOutput(input)
    if self.train then
      local bs = input:size(1)
      local patchsize = input:size(3)
      local flip_mask = torch.randperm(bs):le(bs/2)
      for i=1,input:size(1) do
        if flip_mask[i] == 1 then 
            local powfac = torch.rand(1)[1] * 3.75 + 0.25
            local mulfac = torch.rand(1)[1] * 0.7 + 0.7
            local addval = torch.rand(1)[1] * 0.2 - 0.1
            input[i] = image.rgb2hsv(input[i])
            input[i][{{2,3}, {}, {}}]:pow(powfac):mul(mulfac):add(addval)
            for it = 2, 3 do
               local max = input[i][it]:max()
               local min = input[i][it]:min()
               if min < 0 then
                  input[i][it]:add(-min)
               end
               if max > 1 then
                  input[i][it]:div(max)
               end
            end

         end
      end
    end
    self.output:set(input)
    return self.output
 end
end







