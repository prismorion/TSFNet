namespace TSFNet.Models.MLP
{
    /// <summary>
    /// Снимок параметров MLP (веса и сдвиги) для сохранения и восстановления лучшего состояния сети.
    /// </summary>
    public class MLPSnapshot
    {
        public double[][][] weights;   // веса
        public double[][] biases;      // веса сдвигов

        /// <summary>
        /// Выделение массивов весов и сдвигов под размеры слоёв.
        /// </summary>
        public MLPSnapshot(int[] layerSize)
        {
            // инициализация массивов
            weights = new double[layerSize.Length - 1][][];
            biases = new double[layerSize.Length - 1][];

            // заполнение массивов
            int i = 0;
            for (; i < layerSize.Length - 1; i++)
            {
                biases[i] = new double[layerSize[i + 1]];

                // для каждого нейрона слоя i + 1 хранятся веса от всех нейронов слоя i
                weights[i] = new double[layerSize[i + 1]][];
                for (int j = 0; j < layerSize[i + 1]; j++)
                    weights[i][j] = new double[layerSize[i]];
            }
        }
    }
}
