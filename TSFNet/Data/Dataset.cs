namespace TSFNet.Data
{
    public class Dataset<TInput>
    {
        private TInput[] inputs;
        private double[][] targets;
        private int[] order;
        private static Random rand = new Random();

        public Dataset(TInput[] inputs, double[][] targets)
        {
            this.inputs = inputs;
            this.targets = targets;
            order = new int[inputs.Length];
            ResetOrder();
        }

        public int Length => inputs.Length;

        public TInput GetInput(int i) => inputs[order[i]];
        public double[] GetTarget(int i) => targets[order[i]];
        public TInput[] GetRawInput => inputs;
        public double[][] GetRawTarget => targets;

        public void ResetOrder()
        {
            for (int i = 0; i < inputs.Length; i++)
                order[i] = i;
        }

        public void Shuffle()
        {
            for (int i = order.Length - 1; i > 0; i--)
            {
                int j = rand.Next(i + 1);
                (order[i], order[j]) = (order[j], order[i]);
            }
        }
    }
}
