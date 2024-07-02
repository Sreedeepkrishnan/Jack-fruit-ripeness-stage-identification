function [ im ] = readPathoImage_299( imageFile)

finalDim = 299;

im = imread(imageFile);

% resize image
if size(im,1) ~= finalDim || size(im,2) ~= finalDim
im = imresize(im,[finalDim,finalDim]);
end

end
