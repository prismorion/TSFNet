namespace TSFNet.Training.Parameters
{
    public class TrainingOptions
    {
        public int epochs { get; set; }
        public int reportEvery { get; set; }
        public int patience { get; set; }

        public TrainingOptions(int _epochs = 100, int _reportEvery = 0, int _patience = 20)
        {
            epochs = _epochs;
            reportEvery = _reportEvery;
            patience = _patience;
        }
    }
}
