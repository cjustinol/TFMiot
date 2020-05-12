% Divide la data almacena en ads en datos de entrenamiento, validacion y
% prueba basados en la lista de archivos de validacion y prueba en la
% carpeta de datos datafolder.

function [adsTrain,adsValidation,adsTest] = splitData(ads,datafolder)

% Lee la lista del archivo de validacion
c = fileread(fullfile(datafolder,'validation_list.txt'));
filesValidation = string(split(c));
filesValidation  = filesValidation(filesValidation ~= "");

% Lee la lista del archivo de pruebas
c = fileread(fullfile(datafolder,'testing_list.txt'));
filesTest = string(split(c));
filesTest  = filesTest(filesTest ~= "");

% Determina cuales archivos en el datastore deben ir al conjunto de
% validacion y cuales debe ir al conjunto de prueba.
files = ads.Files;
sf    = split(files,filesep);
isValidation = ismember(sf(:,end-1) + "/" + sf(:,end),filesValidation);
isTest       = ismember(sf(:,end-1) + "/" + sf(:,end),filesTest);

adsTest = subset(ads,isTest);
adsValidation = subset(ads,isValidation);
adsTrain = subset(ads,~isValidation & ~isTest);

end