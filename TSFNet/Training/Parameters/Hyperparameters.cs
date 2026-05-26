namespace TSFNet.Training.Parameters
{
    public class Hyperparameters
    {
        public double learningRate { get; set; }
        public int batchSize { get; set; }
        public double l2Lambda { get; set; }
        public double threshold { get; set; }

        public Hyperparameters(double _learningRate = 0.01, int _batchSize = 1, double _l2Lambda = 0, double _threshold = 5)
        {
            learningRate = _learningRate;
            batchSize = _batchSize;
            l2Lambda = _l2Lambda;
            threshold = _threshold;
        }
    }
}
