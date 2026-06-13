using TSFNet.Calculations;
using TSFNet.Data;
using TSFNet.Training;
using TSFNet.Training.Parameters;

namespace TSFNet.Models.MLP
{
    /// <summary>
    /// Многослойный перцептрон (полносвязная сеть прямого распространения)
    /// с активацией Leaky ReLU в скрытых слоях и линейным выходным слоём.
    /// </summary>
    public class MLP : ITrainable<double[], MLPBuffers, MLPSnapshot>
    {
        private double[][][] weights;   // веса
        private double[][] biases;      // веса сдвигов
        private double[][] layers;      // слои нейронов

        /// <summary>
        /// Создание сети по размерам слоёв; выделение весов/сдвигов и инициализация весов методом He.
        /// </summary>
        public MLP(int[] layerSize)
        {
            // инициализация массивов
            weights = new double[layerSize.Length - 1][][];
            biases = new double[layerSize.Length - 1][];
            layers = new double[layerSize.Length][];

            // заполнение массивов
            int i = 0;
            for (; i < layerSize.Length - 1; i++)
            {
                layers[i] = new double[layerSize[i]];

                biases[i] = new double[layerSize[i + 1]];

                // для каждого нейрона слоя i + 1 хранятся веса от всех нейронов слоя i
                weights[i] = new double[layerSize[i + 1]][];
                for (int j = 0; j < layerSize[i + 1]; j++)
                    weights[i][j] = new double[layerSize[i]];
            }
            layers[i] = new double[layerSize[i]];

            // инициализация весов
            for (int l = 0; l < layers.Length - 1; l++)
            {
                int fanIn = layers[l].Length;
                WeightInitializer.He(weights[l], fanIn, 0.01);
            }
        }

        /// <summary>
        /// Прямой проход: вычисление выхода сети по входному вектору.
        /// </summary>
        public double[] Predict(double[] input)
        {
            Array.Copy(input, layers[0], input.Length);

            // вычисление значений скрытых слоёв
            int i = 1;
            for (; i < layers.Length - 1; i++)
            {
                LinAlgMath.MulMatVec(weights[i - 1], layers[i - 1], layers[i]);
                LinAlgMath.AddVec(layers[i], biases[i - 1], layers[i]);
                ActivationFunctions.LReLU(layers[i], layers[i]);
            }

            // вычисление значений выходного слоя (отсутствует функция активации)
            LinAlgMath.MulMatVec(weights[i - 1], layers[i - 1], layers[i]);
            LinAlgMath.AddVec(layers[i], biases[i - 1], layers[i]);

            return (double[])layers[i].Clone();
        }

        /// <summary>
        /// Обучение на одной эпохе: проход по батчам с накоплением градиентов и обновлением весов.
        /// </summary>
        public void Train(Dataset<double[]> dataset, Hyperparameters options, MLPBuffers mlpBuffers)
        {
            // проход по примерам с шагом в размер батча
            for (int i = 0; i < dataset.Length; i += options.batchSize)
            {
                // выбор длины батча: batchSize или оставшийся хвост
                int size = Math.Min(options.batchSize, dataset.Length - i);

                // проход по батчу
                for (int b = 0; b < size; b++)
                {
                    double[] output = Predict(dataset.GetInput(i + b));
                    LossFunctions.MSEDerivative(dataset.GetTarget(i + b), output, mlpBuffers.dOutputs[b]);

                    // deepcopy слоёв в каждом батче
                    for (int j = 0; j < layers.Length; j++)
                        Array.Copy(layers[j], mlpBuffers.layersHistory[b][j], layers[j].Length);
                }

                Backward(mlpBuffers, size, options.learningRate, options.l2Lambda);
            }
        }

        /// <summary>
        /// Обратное распространение ошибки по батчу и обновление весов и сдвигов (с L2-регуляризацией).
        /// </summary>
        private void Backward(MLPBuffers mlpBuffers, int size, double learningRate, double l2Lambda)
        {
            // обнуление аккумуляторов сумм перед батчем
            mlpBuffers.ClearAccum();

            // проход по примерам батча
            for (int b = 0; b < size; b++)
            {
                int last = weights.Length - 1;
                Array.Copy(mlpBuffers.dOutputs[b], mlpBuffers.deltas[last], mlpBuffers.dOutputs[b].Length);

                for (int i = last; i >= 0; i--)
                {
                    LinAlgMath.MulOuterVec(mlpBuffers.deltas[i], mlpBuffers.layersHistory[b][i], mlpBuffers.DWeights[i]);
                    LinAlgMath.AddMat(mlpBuffers.sumDWeights[i], mlpBuffers.DWeights[i], mlpBuffers.sumDWeights[i]);
                    LinAlgMath.AddVec(mlpBuffers.sumDBiases[i], mlpBuffers.deltas[i], mlpBuffers.sumDBiases[i]);

                    // распространение градиента на слой раньше
                    if (i >= 1)
                    {
                        LinAlgMath.MulMatTVec(weights[i], mlpBuffers.deltas[i], mlpBuffers.deltas[i - 1]);
                        ActivationFunctions.LReLUDerivative(mlpBuffers.layersHistory[b][i], mlpBuffers.derivative[i]);
                        LinAlgMath.MulElemVec(mlpBuffers.deltas[i - 1], mlpBuffers.derivative[i], mlpBuffers.deltas[i - 1]);
                    }
                }
            }

            // апдейт весов и сдвигов
            for (int i = 0; i < weights.Length; i++)
            {
                if (l2Lambda > 0)
                    LinAlgMath.MulMatNum(weights[i], 1.0 - learningRate * l2Lambda, weights[i]);

                LinAlgMath.MulMatNum(mlpBuffers.sumDWeights[i], learningRate / size, mlpBuffers.sumDWeights[i]);
                LinAlgMath.SubMat(weights[i], mlpBuffers.sumDWeights[i], weights[i]);

                LinAlgMath.MulVecNum(mlpBuffers.sumDBiases[i], learningRate / size, mlpBuffers.sumDBiases[i]);
                LinAlgMath.SubVec(biases[i], mlpBuffers.sumDBiases[i], biases[i]);
            }
        }

        /// <summary>
        /// Создание буферов для обучения под текущую конфигурацию сети.
        /// </summary>
        public MLPBuffers CreateBuffer(Dataset<double[]> dataset, Hyperparameters options)
            => new MLPBuffers(layers, weights, biases, options.batchSize);

        /// <summary>
        /// Создание буфера-снимка для сохранения параметров сети.
        /// </summary>
        public MLPSnapshot CreateSnapshotBuffer()
        {
            int[] layersSize = new int[layers.Length];
            for(int i = 0; i < layers.Length; i++)
                layersSize[i] = layers[i].Length;

            return new MLPSnapshot(layersSize);
        }

        /// <summary>
        /// Сохранение текущих весов и сдвигов в снимок.
        /// </summary>
        public void SaveSnapshot(MLPSnapshot mlpSnapshot)
        {
            ArrayCopyUtils.Copy(weights, mlpSnapshot.weights);
            ArrayCopyUtils.Copy(biases, mlpSnapshot.biases);
        }

        /// <summary>
        /// Восстановление весов и сдвигов из снимка.
        /// </summary>
        public void RestoreSnapshot(MLPSnapshot mlpSnapshot)
        {
            ArrayCopyUtils.Copy(mlpSnapshot.weights, weights);
            ArrayCopyUtils.Copy(mlpSnapshot.biases, biases);
        }
    }
}
