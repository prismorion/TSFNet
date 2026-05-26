namespace TSFNet.Models.MLP
{
    public class MLPBuffers
    {
        // буферы распространяются на батч
        public double[][] dOutputs;         // loss
        public double[][][] layersHistory;    // клоны слоёв
        public double[][][] DWeights;       // временная матрица для весов
        public double[][][] sumDWeights;    // накопление градиента весов
        public double[][] sumDBiases;       // накопление градиентов сдвигов
        public double[][] deltas;           // дельты
        public double[][] derivative;       // производные

        public MLPBuffers(double[][] layers, double[][][] weights, double[][] biases, int batchSize)
        {
            // буфер для хранения loss на каждом батче
            int outputSize = layers[^1].Length;
            dOutputs = new double[batchSize][];
            for (int b = 0; b < batchSize; b++)
                dOutputs[b] = new double[outputSize];

            // буфер для клонирования слоёв на каждом батче
            layersHistory = new double[batchSize][][];
            for (int b = 0; b < batchSize; b++)
            {
                layersHistory[b] = new double[layers.Length][];
                for (int l = 0; l < layers.Length; l++)
                    layersHistory[b][l] = new double[layers[l].Length];
            }

            // аккумуляторы градиентов на батч и временная матрица для весов
            DWeights = new double[weights.Length][][];
            sumDWeights = new double[weights.Length][][];
            sumDBiases = new double[biases.Length][];
            for (int i = 0; i < weights.Length; i++)
            {
                DWeights[i] = new double[weights[i].Length][];
                sumDWeights[i] = new double[weights[i].Length][];
                for (int j = 0; j < weights[i].Length; j++)
                {
                    DWeights[i][j] = new double[weights[i][j].Length];
                    sumDWeights[i][j] = new double[weights[i][j].Length];
                }
                sumDBiases[i] = new double[biases[i].Length];
            }

            // буфер для вычисленной delta на каждом слое
            deltas = new double[layers.Length - 1][];
            for (int i = 0; i < layers.Length - 1; i++)
                deltas[i] = new double[layers[i + 1].Length];

            // буфер для вычисления производной
            derivative = new double[layers.Length][];
            for (int i = 0; i < layers.Length; i++)
                derivative[i] = new double[layers[i].Length];
        }

        public void ClearAccum()
        {
            for (int i = 0; i < sumDWeights.Length; i++)
            {
                for (int j = 0; j < sumDWeights[i].Length; j++)
                    Array.Clear(sumDWeights[i][j]);
                Array.Clear(sumDBiases[i]);
            }
        }
    }
}
