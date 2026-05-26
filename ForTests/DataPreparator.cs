namespace ForTests
{
    public static class DataPreparator
    {
        public static double[][] InputsPreparation(List<Dot> records, int windowSize, int output)
        {
            int samples = (records.Count - windowSize) / output;
            double[][] inputs = new double[samples][];

            for (int i = 0; i < samples; i++)
            {
                inputs[i] = new double[windowSize];
                int start = i * output;
                for (int j = 0; j < windowSize; j++)
                    inputs[i][j] = records[start + j].Y;
            }

            return inputs;
        }

        public static double[][][] SeqInputsPreparation(List<Dot> records, int windowSize, int output)
        {
            int samples = (records.Count - windowSize) / output;
            double[][][] seqInputs = new double[samples][][];

            for (int i = 0; i < samples; i++)
            {
                seqInputs[i] = new double[windowSize][];
                int start = i * output;
                for (int j = 0; j < windowSize; j++)
                    seqInputs[i][j] = [records[start + j].Y];
            }

            return seqInputs;
        }

        public static double[][] TargetsPreparation(List<Dot> records, int windowSize, int output)
        {
            int samples = (records.Count - windowSize) / output;
            double[][] targets = new double[samples][];

            for (int i = 0; i < samples; i++)
            {
                targets[i] = new double[output];
                int start = i * output;
                for (int j = 0; j < output; j++)
                    targets[i][j] = records[start + windowSize + j].Y;
            }

            return targets;
        }
    }
}
