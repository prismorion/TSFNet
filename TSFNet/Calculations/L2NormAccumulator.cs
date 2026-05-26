namespace TSFNet.Calculations
{
    public class L2NormAccumulator
    {
        private double sumSq = 0;

        public void AddVectors(params double[][] vector)
        {
            for (int i = 0; i < vector.Length; i++)
                for (int j = 0; j < vector[i].Length; j++)
                    sumSq += vector[i][j] * vector[i][j];
        }

        public void AddMatrices(params double[][][] matrix)
        {
            for (int i = 0; i < matrix.Length; i++)
                for (int j = 0; j < matrix[i].Length; j++)
                    for (int k = 0; k < matrix[i][j].Length; k++)
                        sumSq += matrix[i][j][k] * matrix[i][j][k];
        }

        public double GetNorm() => Math.Sqrt(sumSq);

        public void Reset()
        {
            sumSq = 0;
        }
    }
}
