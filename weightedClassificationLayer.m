classdef weightedClassificationLayer < nnet.layer.ClassificationLayer
    
    properties
        % Vector de fila de pesos correspondientes a las clases en los
        % datos de entrenamiento
        ClassWeights
    end
    
    methods
        function layer = weightedClassificationLayer(classWeights, name)
            % crea la capa de weighted cross entropy loss. 
            % classWeights es un vector de fila de pesos correspondientes a
            % las clases en el orden que aparecen en la data de
            % entrenamiento.
            
            % Se establece los pesos de la clase
            layer.ClassWeights = classWeights;
            
            % Se establece el nombre de la capa
            if nargin == 2
                layer.Name = name;
            end
            
            % Se establece la descripcion de la capa
            layer.Description = 'Weighted cross entropy';
        end
        
        function loss = forwardLoss(layer, Y, T)
            % retorna el weighted cross entropy losse entre las Y y los
            % objetivos de entrenamiento T.
            
            N = size(Y,4);
            Y = squeeze(Y);
            T = squeeze(T);
            W = layer.ClassWeights;
    
            loss = -sum(W*(T.*log(Y)))/N;
        end
        
        function dLdY = backwardLoss(layer, Y, T)
            % devuelve las derivadas del weighted cross entropy loss con
            % respecto a las predicciones Y
            
            [~,~,K,N] = size(Y);
            Y = squeeze(Y);
            T = squeeze(T);
            W = layer.ClassWeights;
            
            dLdY = -(W'.*T./Y)/N;
            dLdY = reshape(dLdY,[1 1 K N]);
        end
    end
end

