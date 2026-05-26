namespace TSFNet.Training.Responses
{
    public class FitResponse
    {
        public double timeElapsed;
        public Dictionary<int, double> logTrainLoss;

        public FitResponse(double _timeElapsed, Dictionary<int, double> _logTrainLoss)
        {
            timeElapsed = _timeElapsed;
            logTrainLoss = _logTrainLoss;
        }
    }
}
