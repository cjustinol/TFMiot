close all; clear all;

%% Canal Sostenibilidad ambiental: motosierra
ChannelIDParking = 981480;
readAPIKeyParking = '02IM3N7ITH0JJOK1';
writeAPIKeyParking = 'EUT4ONWZVWPMUK8V';

load trainedNet

%% Clasificar sonidos a partir de un audio generado
[Datos,fs]=audioread('RuidosVarios (formado uniendo varios audios).wav');

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

% Constantes de control
epsil = 1e-6;
Ndetecciones = 0;
NdeteccionesPrevias = 0;

%Definimos un buffer para extraer tramos de datos con fines de clasificacion 
LongitudBuffer = fs/3;
% Bucle de clasificacion
iteraciones = floor(size(Datos,1)/LongitudBuffer);

%% Alarma para ThingSpeak
AlarmaThingSpeak = 0;
dataField1 = AlarmaThingSpeak;
respuesta = thingSpeakWrite(ChannelIDParking,dataField1,'Fields',1,'Writekey',writeAPIKeyParking);
pause(2); pause(2); pause(2); pause(2); pause(2); pause(2); pause(2); pause(2); pause(2); pause(2); % > 16 segundos  

% Inicializacion barra de progreso
% for i=1:iteraciones
for i=1:200
  x = Datos((i-1)*LongitudBuffer+1:i*LongitudBuffer); % Wave buffer 

  % Reproduccion del audio desde el ordenador
  % sound(x,fs);    
        
  numSamples = size(x,1);

  numToPadFront = floor( (segmentSamples - numSamples)/2 );
  numToPadBack = ceil( (segmentSamples - numSamples)/2 );
  
  xPadded = [zeros(numToPadFront,1,'like',x);x;zeros(numToPadBack,1,'like',x)];

  %% Se extrae las caracteristicas del audio. La salida es un Bark spectrum con tiempo entre filas.
  features = extract(afe,xPadded);
  [numHops,numFeatures] = size(features);

  %% audioFeatureExtractor normaliza los espectrogramas auditivos por la potencia de la ventana para que las mediciones sean independientes del tipo de ventana y la duración de la ventana. Se post-procesa el espectrograma auditivo aplicando un logaritmo. Tomar un registro de números pequeños puede conducir a un error de redondeo. Para evitar el error de redondeo, se invertirá la normalización de la ventana.
  % Determina el factor de desnormalizacion a aplicar.
  unNorm = 2/(sum(afe.Window)^2);
 
  xPadded = [zeros(floor((segmentSamples-size(x,1))/2),1);x;zeros(ceil((segmentSamples-size(x,1))/2),1)];
  X(:,:) = extract(afe,xPadded);
  
  %% Escala las caracteristicas con el window power y luego toma el log. Para obtener datos con una distribución más uniforme, se toma el logaritmo de los espectrogramas utilizando un pequeño desplazamiento.
  XdNor = X/unNorm;
  epsil = 1e-6;
  
  % Espectrograma de clasificacion
  XClasificacion = log10(XdNor + epsil);

  % Clasifica el espectrograma actual, save the label to the label buffer,
  [YPredicted,probs] = classify(trainedNet,XClasificacion,"ExecutionEnvironment",'cpu');
  %figure(1); imagesc(XClasificacion'); title(YPredicted);
  %pause(2);pause(2);pause(2);
  
  if YPredicted == 'motosierra'
    %disp(YPredicted);
    figure(1); imagesc(XClasificacion'); title('Motosierra')
    %% Para la ejecución en el movil esta linea hay que comentarla por no tener salida audio
    % sound(XClasificacion,fs);
    
    % La deteccion de un sonido de motosierra incrementa el valor
    Ndetecciones = Ndetecciones + 1;
  end

 if mod(Ndetecciones,25)==0 && (Ndetecciones > 0) && Ndetecciones > NdeteccionesPrevias 
    NdeteccionesPrevias = Ndetecciones;
    % Escribir las detecciones acumuladas en Field2 y manejar Alarma en
    % ThingSpeak
    AlarmaThingSpeak = 1;
    dataField3 = AlarmaThingSpeak;
    respuesta = thingSpeakWrite(ChannelIDParking,dataField3,'Fields',3,'Writekey',writeAPIKeyParking);
    %pause(15); %Esperar a consolidar la escritura, hay que hacer pausas de 2 segundos en Drive hasta un total de 15 segundos
    pause(2); pause(2); pause(2); pause(2); pause(2); pause(2); pause(2); pause(2); pause(2); pause(2); % > 16 segundos  
    
    dataField4 = Ndetecciones;
    respuesta = thingSpeakWrite(ChannelIDParking,dataField4,'Fields',4,'Writekey',writeAPIKeyParking);
    %pause(15); %Esperar a consolidar la escritura
    pause(2); pause(2); pause(2); pause(2); pause(2); pause(2); pause(2); pause(2); pause(2); pause(2); % > 16 segundos  

    AlarmaThingSpeak = 0;
    dataField3 = AlarmaThingSpeak;
    respuesta = thingSpeakWrite(ChannelIDParking,dataField3,'Fields',3,'Writekey',writeAPIKeyParking);
    %pause(15); %Esperar a consolidar la escritura
    pause(2); pause(2); pause(2); pause(2); pause(2); pause(2); pause(2); pause(2); pause(2); pause(2);% > 16 segundos  

  end
end

%% Al final de todo reportamos todas las detecciones
dataField2 = Ndetecciones; 
respuesta = thingSpeakWrite(ChannelIDParking,dataField2,'Fields',2,'Writekey',writeAPIKeyParking);
%pause(15); %Esperar a consolidar la escritura
pause(2); pause(2); pause(2); pause(2); pause(2); pause(2); pause(2); pause(2); pause(2); pause(2); % > 16 segundos  

