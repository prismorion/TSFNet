namespace ForTests
{
    public static class DataPreparator
    {
        public static double[][] InputsPreparation(List<Dot> records, int windowSize, int outputSize)
        {
            int samples = records.Count - windowSize - outputSize + 1;
            double[][] inputs = new double[samples][];
            for (int i = 0; i < samples; i++)
            {
                inputs[i] = new double[windowSize];
                for (int j = 0; j < windowSize; j++)
                    inputs[i][j] = records[i + j].Y;
            }
            return inputs;
        }
        public static double[][][] SeqInputsPreparation(List<Dot> records, int windowSize, int outputSize)
        {
            int samples = records.Count - windowSize - outputSize + 1;
            double[][][] seqInputs = new double[samples][][];
            for (int i = 0; i < samples; i++)
            {
                seqInputs[i] = new double[windowSize][];
                for (int j = 0; j < windowSize; j++)
                    seqInputs[i][j] = [records[i + j].Y];
            }
            return seqInputs;
        }
        public static double[][] TargetsPreparation(List<Dot> records, int windowSize, int outputSize)
        {
            int samples = records.Count - windowSize - outputSize + 1;
            double[][] targets = new double[samples][];
            for (int i = 0; i < samples; i++)
            {
                targets[i] = new double[outputSize];
                for (int j = 0; j < outputSize; j++)
                    targets[i][j] = records[i + windowSize + j].Y;
            }
            return targets;
        }
    }
}