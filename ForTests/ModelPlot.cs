namespace ForTests
{
    public class ModelPlot
    {
        public string Name { get; set; }
        public Color Color { get; set; }
        public double[] X { get; set; }
        public double[] Y { get; set; }

        public ModelPlot(string name, Color color, double[] X, double[] Y)
        {
            Name = name;
            Color = color;
            this.X = X;
            this.Y = Y;
        }
    }
}
