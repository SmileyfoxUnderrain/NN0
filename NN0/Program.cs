using System;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;

namespace NN0
{
    /// <summary>
    /// For example, we have a segment-based digit indicator
    /// and the number of the input neurons equals to the number of 
    /// segments in the indicator.
    ///              _     0
    ///             /_/   123
    ///            /_/   456
    /// And totally 7 neurons on the input layer, which will
    /// take a brightness level of the each segment
    /// On the output we are awaiting for the nn to guess the digit that was indicated:
    /// So, we need 10 neurons on the output layer
    /// We will also add one hidden layer with quantity of 8 neurons on it
    /// 
    /// </summary>
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Hello World!");
            var nn = NeuralNetworkFactory.CreateByLayerSizes(new[] { 7, 8, 10 });
            var epoch = new Epoch();
            epoch.Samples.AddRange(new[] {
                new Sample(new double[] { 1, 1, 0, 1, 1, 1, 1 }, 
                    new double[] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 }), //0

                new Sample(new double[] { 0, 0, 0, 1, 0, 0, 1 }, 
                    new double[] { 0, 0, 0, 0, 0, 0, 0, 0, 1, 0 }), //1

                new Sample(new double[] { 1, 0, 1, 1, 1, 1, 0 }, 
                    new double[] { 0, 0, 0, 0, 0, 0, 0, 1, 0, 0 }), //2

                new Sample(new double[] { 1, 0, 1, 1, 0, 1, 1 }, 
                    new double[] { 0, 0, 0, 0, 0, 0, 1, 0, 0, 0 }), //3

                new Sample(new double[] { 0, 1, 1, 1, 0, 0, 1 }, 
                    new double[] { 0, 0, 0, 0, 0, 1, 0, 0, 0, 0 }), //4

                new Sample(new double[] { 1, 1, 1, 0, 0, 1, 1 }, 
                    new double[] { 0, 0, 0, 0, 1, 0, 0, 0, 0, 0 }), //5

                new Sample(new double[] { 1, 1, 1, 0, 1, 1, 1 }, 
                    new double[] { 0, 0, 0, 1, 0, 0, 0, 0, 0, 0 }), //6

                new Sample(new double[] { 1, 0, 0, 1, 0, 0, 1 }, 
                    new double[] { 0, 0, 1, 0, 0, 0, 0, 0, 0, 0 }), //7

                new Sample(new double[] { 1, 1, 1, 1, 1, 1, 1 }, 
                    new double[] { 0, 1, 0, 0, 0, 0, 0, 0, 0, 0 }), //8

                new Sample(new double[] { 1, 1, 1, 1, 0, 1, 1 }, 
                    new double[] { 1, 0, 0, 0, 0, 0, 0, 0, 0, 0 })  //9
            });
            
            
            for (var i = 0; i < 400; i++)
            {
                Console.WriteLine($"current lesson is {i}");
                while(epoch.HasNextRandomSample)
                {
                    var sample = epoch.GetNextRandomSample();
                    nn.Learn(sample.InputVector, sample.AwaitedResponse);
                }
                epoch.ResetRandomizer();
            }
            // Control Check
            foreach(var sample in epoch.Samples)
            {
                var result = nn.Calculate(sample.InputVector);
                var shortResult = result.Select(r => Math.Round(r, 3));
                Console.WriteLine($"result: {string.Join(", ", shortResult)}");
                Console.WriteLine();
            }
        }
    }
}
