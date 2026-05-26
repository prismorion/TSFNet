namespace TSFNet.Training.Responses
{
    public class FitEarlyStoppingResponse
    {
        public double timeElapsed;
        public int bestEpoch;
        public double bestValidationLoss;
        public Dictionary<int, double> logTrainLoss;
        public Dictionary<int, double> logValidationLoss;

        public FitEarlyStoppingResponse(double _timeElapsed, int _bestEpoch, double _bestValidationLoss,
            Dictionary<int, double> _logTrainLoss, Dictionary<int, double> _logValidationLoss)
        {
            timeElapsed = _timeElapsed;
            bestEpoch = _bestEpoch;
            bestValidationLoss = _bestValidationLoss;
            logTrainLoss = _logTrainLoss;
            logValidationLoss = _logValidationLoss;
        }
    }
}
