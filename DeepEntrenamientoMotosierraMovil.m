close all; clear all;

%% Reconocimiento de audio usando Deep Learning

%% Carga del conjunto de audios
% La variable datafolder contendra la ubicacion de la carpeta con los
% audios. Se usa |audioDatastore| para crear un datastore que contiene los
% nombres de los archivos y las etiquetas correspondientes. Se usa el nombre 
% de los folders como el origen de la etiqueta.
% Se especifica el metodo de lectura para el archivo de audio entero y se
% crea una copia del datastore para luego ser usada.
datafolder = './RuidosReconocimiento';
adsSubset = audioDatastore(datafolder, ...
    'IncludeSubfolders',true, ...
    'FileExtensions','.wav', ...
    'LabelSource','foldernames');
ads0 = copy(adsSubset);

%% Eligiendo las clases a reconocer
% Se especifica las clases que deseamos reconocer
% Se etiqueta todas las palabras que no estan en el grupo de comandos como |desconocido|. 
% Para reducir el desequilibrio de clase entre los comandos y los desconocidos y acelerar el procesamiento, 
% solo se incluye una fracción |includeFraction| de las palabras desconocidas en el conjunto de entrenamiento. 
% Se usa |subset| para crear un almacén de datos que contenga solo el grupo de comandos 
% y el subconjunto de palabras desconocidas. Finalmente se cuenta la cantidad de ejemplos que pertenecen a cada clase.

commands = categorical(["motosierra","fondo"]);

isCommand = ismember(adsSubset.Labels,commands);
isUnknown = ~ismember (adsSubset.Labels,[commands,"_ruido_fondo_"]);

includeFraction = 0.5;
mask = rand(numel(adsSubset.Labels),1) < includeFraction;
isUnknown = isUnknown & mask;
adsSubset.Labels(isUnknown) = categorical("desconocido");

adsSubset = subset(adsSubset,isCommand|isUnknown);
countEachLabel(adsSubset)

%% Divide la data en conjuntos de entrenamiento, validacion y prueba
% La carpeta del dataset contiene archivos de texto, que enumeran los 
% archivos de audio que se utilizarán como conjuntos de validación y prueba.
[adsTrain,adsValidation,adsTest] = splitData(adsSubset,datafolder);

%% Calculo de spectrogramas auditivos
% Se preparar los datos para el entrenamiento de la red neuronal convolucional
%
% Parametro para el calculo del espectrograma
% |segmentDuration| es la duración de cada sppech clip (en segundos).
% |frameDuration| es la duración de cada cuadro para el cálculo del espectrograma. 
% |hopDuration| es el paso de tiempo entre cada columna del espectrograma.
% |numBands| es el numero de filtros log-bark y es igual a la altura de cada espectrograma.
fs = 48e3; % Frecuencia de muestreo conocida

segmentDuration = 1;
frameDuration = 0.02;
hopDuration = 0.01;

segmentSamples = round(segmentDuration*fs);
frameSamples = round(frameDuration*fs);
hopSamples = round(hopDuration*fs);
overlapSamples = frameSamples - hopSamples;

FFTLength = 2*512;
numBands = 40;

afe = audioFeatureExtractor( ...
    'SampleRate',fs, ...
    'FFTLength',FFTLength, ...
    'Window',hann(frameSamples,'periodic'), ...
    'OverlapLength',overlapSamples, ...
    'barkSpectrum',true);
setExtractorParams(afe,'barkSpectrum','NumBands',numBands);

%%
% Lee un archivo del dataset. El entrenamiento de una red neuronal convolucional requiere que la entrada sea de un tamaño consistente. Se aplica zero-pad en la parte frontal y posterior de la señal de audio para que sea de una longitud segmentSamples.

epsil = 1e-6;

x = read(adsTrain);

numSamples = size(x,1);

numToPadFront = floor( (segmentSamples - numSamples)/2 );
numToPadBack = ceil( (segmentSamples - numSamples)/2 );

xPadded = [zeros(numToPadFront,1,'like',x);x;zeros(numToPadBack,1,'like',x)];

 %% Se extrae las caracteristicas del audio. La salida es un Bark spectrum con tiempo entre filas.
features = extract(afe,xPadded);
[numHops,numFeatures] = size(features);

%% audioFeatureExtractor normaliza los espectrogramas auditivos por la potencia de la ventana para que las mediciones sean independientes del tipo de ventana y la duración de la ventana. Se post-procesa el espectrograma auditivo aplicando un logaritmo. Tomar un registro de números pequeños puede conducir a un error de redondeo. Para evitar el error de redondeo, se invertirá la normalización de la ventana.
% Factor de desnormalizacion a aplicar:
unNorm = 2/(sum(afe.Window)^2);

%% Para acelerar el procesamiento, puede distribuir la extracción de características entre varios trabajadores se puede usar el parallel-for-loop.
reduceDataset = true;
if ~isempty(ver('parallel')) && ~reduceDataset
    pool = gcp;
    numPar = numpartitions(adsTrain,pool);
else
    numPar = 1;
end

%% Para cada partición, se lee desde el datastore, rellenando (zero-pad) con cero la señal y luego extrayendo las características.
for ii = 1:numPar
    subds = partition(adsTrain,numPar,ii);
    XTrain = zeros(numHops,numBands,1,numel(subds.Files));
    for idx = 1:numel(subds.Files)
        x = read(subds);
        xPadded = [zeros(floor((segmentSamples-size(x,1))/2),1);x;zeros(ceil((segmentSamples-size(x,1))/2),1)];
        XTrain(:,:,:,idx) = extract(afe,xPadded);
    end
    XTrainC{ii} = XTrain;
end

%% Se convierte la salida a una matriz de 4 dimensiones con espectrogramas auditivos a lo largo de la cuarta dimensión.
XTrain = cat(4,XTrainC{:});

[numHops,numBands,numChannels,numSpec] = size(XTrain);

%% Se escala las características por el poder de la ventana y luego se toma el logaritmo.
XTrain = XTrain/unNorm;
epsil = 1e-6;
XTrain = log10(XTrain + epsil);

%% Se realiza los pasos de extracción de características descritos anteriormente para el conjunto de validación.
if ~isempty(ver('parallel'))&& ~reduceDataset
    pool = gcp;
    numPar = numpartitions(adsValidation,pool);
else
    numPar = 1;
end
for ii = 1:numPar
    subds = partition(adsValidation,numPar,ii);
    XValidation = zeros(numHops,numBands,1,numel(subds.Files));
    for idx = 1:numel(subds.Files)
        x = read(subds);
        xPadded = [zeros(floor((segmentSamples-size(x,1))/2),1);x;zeros(ceil((segmentSamples-size(x,1))/2),1)];
        XValidation(:,:,:,idx) = extract(afe,xPadded);
    end
    XValidationC{ii} = XValidation;
end
XValidation = cat(4,XValidationC{:});
XValidation = XValidation/unNorm;
XValidation = log10(XValidation + epsil);

%% Se aislan las etiquetas de entrenamiento y validacion y se eliminan las categorias vacias
YTrain = removecats(adsTrain.Labels);
YValidation = removecats(adsValidation.Labels);

%% Visualizando la Data
%Se traza las formas de onda y los espectrogramas auditivos de algunas muestras de entrenamiento. Tambien se reproduce los clips de audio correspondientes.
specMin = min(XTrain,[],'all');
specMax = max(XTrain,[],'all');
idx = randperm(numel(adsTrain.Files),3);
figure('Units','normalized','Position',[0.2 0.2 0.6 0.6]);
for i = 1:3
    [x,fs] = audioread(adsTrain.Files{idx(i)});
    subplot(2,3,i)
    plot(x)
    axis tight
    title(string(adsTrain.Labels(idx(i))))

    subplot(2,3,i+3)
    spect = (XTrain(:,:,1,idx(i))');
    pcolor(spect)
    caxis([specMin specMax])
    shading flat

    sound(x,fs)
    pause(2)
end

%% Añadiendo datos de ruido como brackground
% Se usa los archivos de audio en la carpeta _ruido_fondo_ para crear muestras de clips de un segundo de ruido de fondo. 
% Se crea un número igual de clips de fondo de cada archivo de ruido de fondo.
% Antes de calcular los espectrogramas, la función cambia la escala de cada clip de audio con un factor muestreado de una distribución logarítmica uniforme en el rango dado por volumeRange.

adsBkg = subset(adsSubset,adsSubset.Labels=="_ruido_fondo_");
 numBkgClips = 4000;
if reduceDataset
    numBkgClips = numBkgClips/20;
end
volumeRange = log10([1e-4,1]);

numBkgFiles = numel(adsBkg.Files);
numClipsPerFile = histcounts(1:numBkgClips,linspace(1,numBkgClips,numBkgFiles+1));
Xbkg = zeros(size(XTrain,1),size(XTrain,2),1,numBkgClips,'single');
bkgAll = readall(adsBkg);
ind = 1;

for count = 1:numBkgFiles
    bkg = bkgAll{count};
    idxStart = randi(numel(bkg)-fs,numClipsPerFile(count),1);
    idxEnd = idxStart+fs-1;
    gain = 10.^((volumeRange(2)-volumeRange(1))*rand(numClipsPerFile(count),1) + volumeRange(1));
    for j = 1:numClipsPerFile(count)

        x = bkg(idxStart(j):idxEnd(j))*gain(j);

        x = max(min(x,1),-1);

        Xbkg(:,:,:,ind) = extract(afe,x);

        if mod(ind,1000)==0
            disp("Processed " + string(ind) + " background clips out of " + string(numBkgClips))
        end
        ind = ind + 1;
    end
end
Xbkg = Xbkg/unNorm;
Xbkg = log10(Xbkg + epsil);

%% Se divide los espectrogramas de ruido de fondo entre los conjuntos de entrenamiento, validacion y prueba. Debido a  la que carpeta _ruido_fondo contiene solo unos 5 minutos y medio de ruido de fondo, las muestras de fondo en los diferentes conjuntos de datos estan altamente correlacionadas.  
numTrainBkg = floor(0.85*numBkgClips);
numValidationBkg = floor(0.15*numBkgClips);

XTrain(:,:,:,end+1:end+numTrainBkg) = Xbkg(:,:,:,1:numTrainBkg);
YTrain(end+1:end+numTrainBkg) = "fondo";

XValidation(:,:,:,end+1:end+numValidationBkg) = Xbkg(:,:,:,numTrainBkg+1:end);
YValidation(end+1:end+numValidationBkg) = "fondo";

%% Se grafica la distribución de las diferentes etiquetas de clase en los conjuntos de entrenamiento y validación.
figure('Units','normalized','Position',[0.2 0.2 0.5 0.5])

subplot(2,1,1)
histogram(YTrain)
title("Distribucion Etiquetas Entrenamiento")

subplot(2,1,2)
histogram(YValidation)
title("Distribucion Etiquetas Validacion")

%% Arquitectura de la red neuronal

classWeights = 1./countcats(YTrain);
classWeights = classWeights'/mean(classWeights);
numClasses = numel(categories(YTrain));

timePoolSize = ceil(numHops/8);

dropoutProb = 0.2;
numF = 12;
layers = [
    imageInputLayer([numHops numBands])

    convolution2dLayer(3,numF,'Padding','same')
    batchNormalizationLayer
    reluLayer

    maxPooling2dLayer(3,'Stride',2,'Padding','same')

    convolution2dLayer(3,2*numF,'Padding','same')
    batchNormalizationLayer
    reluLayer

    maxPooling2dLayer(3,'Stride',2,'Padding','same')

    convolution2dLayer(3,4*numF,'Padding','same')
    batchNormalizationLayer
    reluLayer

    maxPooling2dLayer(3,'Stride',2,'Padding','same')

    convolution2dLayer(3,4*numF,'Padding','same')
    batchNormalizationLayer
    reluLayer
    convolution2dLayer(3,4*numF,'Padding','same')
    batchNormalizationLayer
    reluLayer

    maxPooling2dLayer([timePoolSize,1])

    dropoutLayer(dropoutProb)
    fullyConnectedLayer(numClasses)
    softmaxLayer
    weightedClassificationLayer(classWeights)];

%% Entrenamiento de la red
% Se especifica las opciones de entrenamiento. Se hace uso del optimizador
% Adam, se entrena durante 25 epocas y se reduce la taza de aprendizaje en un
% factor de 10 despues de 20 epocas.
miniBatchSize = 128;
validationFrequency = floor(numel(YTrain)/miniBatchSize);
options = trainingOptions('adam', ...
    'InitialLearnRate',3e-4, ...
    'MaxEpochs',25, ...
    'MiniBatchSize',miniBatchSize, ...
    'Shuffle','every-epoch', ...
    'Plots','training-progress', ...
    'Verbose',false, ...
    'ValidationData',{XValidation,YValidation}, ...
    'ValidationFrequency',validationFrequency, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',0.1, ...
    'LearnRateDropPeriod',20);

%% Se entrena la red. 
trainedNet = trainNetwork(XTrain,YTrain,layers,options);

%% Se evelua la red entrenada
% Calcula la precisión final de la red en el conjunto de entrenamiento (sin
% aumento de datos) y el conjunto de validación. La red es muy precisa en
% este conjunto de datos. Sin embargo, los datos de entrenamiento,
% validación y prueba tienen distribuciones similares que no necesariamente
% reflejan entornos del mundo real. Esta limitación se aplica
% particularmente a la categoría desconocida, que contiene un pequeño
% conjunto de audios de relacionados a sonidos de television y trafico.

%if reduceDataset
%    load('commandNet.mat','trainedNet');
%end

YValPred = classify(trainedNet,XValidation);
validationError = mean(YValPred ~= YValidation);
YTrainPred = classify(trainedNet,XTrain);
trainError = mean(YTrainPred ~= YTrain);
disp("Error Entrenamiento: " + trainError*100 + "%")
disp("Error Validacion: " + validationError*100 + "%")

%% Despliegue de la matriz de confusion.
% Se muestra la precisión y el recall de cada clase mediante el uso de resúmenes de columnas y filas
figure('Units','normalized','Position',[0.2 0.2 0.5 0.5]);
cm = confusionchart(YValidation,YValPred);
cm.Title = 'Matiz Confusion Validacion Datos';
cm.ColumnSummary = 'column-normalized';
cm.RowSummary = 'row-normalized';
sortClasses(cm, [commands,"desconocido"])

% Se realiza el entrenamiento en el cpu debido a los recurso limitados del
% movil.

info = whos('trainedNet');
disp("Dimensión de la Red: " + info.bytes/1024 + " kB")

for i = 1:100
    x = randn([numHops,numBands]);
    tic
    [YPredicted,probs] = classify(trainedNet,x,"ExecutionEnvironment",'cpu');
    time(i) = toc;
end
disp("Tiempo de predicción una imagen en CPU: " + mean(time(11:end))*1000 + " ms")

save trainedNet trainedNet