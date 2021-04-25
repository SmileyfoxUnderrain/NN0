using System;
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
    /// For the first task, our nn will try to learn to guess if the digit indicated 
    /// is even or odd, so we need 7-4-1 network. To guess that indicated "7" is an odd number,
    /// we will send an 1-0-0-1-0-0-1 signal and will await for 1 on the end
    /// </summary>
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Hello World!");
            var nn = NeuralNetworkFactory.CreateByLayerSizes(new[] { 7, 1 });
            
            for (var i = 0; i < 300; i++)
            {
                Console.WriteLine($"current lesson is {i}");

                //0 is even
                nn.Learn(new double[] { 1, 1, 0, 1, 1, 1, 1 }, new double[] { 0 });
                //1 is odd
                nn.Learn(new double[] { 0, 0, 0, 1, 0, 0, 1 }, new double[] { 1 });
                // 2 is even
                nn.Learn(new double[] { 1, 0, 1, 1, 1, 1, 0 }, new double[] { 0});
                //// 3 is odd
                nn.Learn(new double[] { 1, 0, 1, 1, 0, 1, 1 }, new double[] { 1 });
                //// 4 is even
                nn.Learn(new double[] { 0, 1, 1, 1, 0, 0, 1 }, new double[] { 0 });
                //// 5 is odd
                nn.Learn(new double[] { 1, 1, 1, 0, 0, 1, 1 }, new double[] { 1 });
                //// 6 is even
                nn.Learn(new double[] { 1, 1, 1, 0, 1, 1, 1 }, new double[] { 0 });
                //// 7 is odd
                nn.Learn(new double[] { 1, 0, 0, 1, 0, 0, 1 }, new double[] { 1 });
                //// 8 is even
                nn.Learn(new double[] { 1, 1, 1, 1, 1, 1, 1 }, new double[] { 0 });
                //// 9 is odd
                nn.Learn(new double[] { 1, 1, 1, 1, 0, 1, 1 }, new double[] { 1 });
            }
            // Control Check
            var result0 = nn.Calculate(new double[] { 1, 1, 0, 1, 1, 1, 1 });
            Console.WriteLine($"0 is {string.Join(", ", result0)}");

            var result1 = nn.Calculate(new double[] { 0, 0, 0, 1, 0, 0, 1 });
            Console.WriteLine($"1 is {string.Join(", ", result1)}");

            var result2 = nn.Calculate(new double[] { 1, 0, 1, 1, 1, 1, 0 });
            Console.WriteLine($"2 is {string.Join(", ", result2)}");

            var result3 = nn.Calculate(new double[] { 1, 0, 1, 1, 0, 1, 1 });
            Console.WriteLine($"3 is {string.Join(", ", result3)}");

            var result4 = nn.Calculate(new double[] { 0, 1, 1, 1, 0, 0, 1 });
            Console.WriteLine($"4 is {string.Join(", ", result4)}");

            var result5 = nn.Calculate(new double[] { 1, 1, 1, 0, 0, 1, 1 });
            Console.WriteLine($"5 is {string.Join(", ", result5)}");

            var result6 = nn.Calculate(new double[] { 1, 1, 1, 0, 1, 1, 1 });
            Console.WriteLine($"6 is {string.Join(", ", result6)}");

            var result7 = nn.Calculate(new double[] { 1, 0, 0, 1, 0, 0, 1 });
            Console.WriteLine($"7 is {string.Join(", ", result7)}");

            var result8 = nn.Calculate(new double[] { 1, 1, 1, 1, 1, 1, 1 });
            Console.WriteLine($"8 is {string.Join(", ", result8)}");

            var result9 = nn.Calculate(new double[] { 1, 1, 1, 1, 0, 1, 1 });
            Console.WriteLine($"9 is {string.Join(", ", result9)}");

        }
    }
}
