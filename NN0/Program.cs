using NN0.ActivationFunctions;
using NN0.LossFunctions;
using System;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;

namespace NN0
{

    class Program
    {
        static void Main(string[] args)
        {
            BuildNnForCathegorizingProblems();
            //PointsOnThePlaneOverfitting();
            //CelsiusToFarenheit();
            //SegmentDigits();
        }
        private static void BuildNnForCathegorizingProblems()
        {
            var factory = new NeuralNetworkFactory();
            var nn = factory
                .PrepareNetwork(0.1)
                .SetInputLayer(7, true)
                .AddLayer(ActivationFunctionType.Logistic, 7, true)
                .AddLayer(ActivationFunctionType.Softmax, 10, false, true)
                .GetPreparedNetwork();

            var selection = new Selection();
            selection.Samples.AddRange(new[] {
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

            nn.TrainWithSelection(selection, 400);
            nn.CheckWithSelection(selection);

        }
        /// <summary>
        /// The probleb shows how overfitting occurs in NN with excess number of 
        /// neurons.
        /// On the hidden layer we will use the linear activation function.
        /// Because it's the classification problem, 
        /// it will be better to use the activation function with an output in the range -1, 1
        /// on the output layer: There are tahn and softsign.
        /// The hinge and squared hinge are better situable to be used as a loss function
        /// to evaluate the quality criterion.
        /// The overfitting happens after ~200 cycles
        /// </summary>
        private static void PointsOnThePlaneOverfitting()
        {
            var nn = NeuralNetworkFactory
                .CreateByFunctionTypeAndLayerSizes(ActivationFunctionType.Identity, new[] { 2, 1, 1, 1, 2 }, 0.01);
            nn.SetOutputActivationFunctionType(ActivationFunctionType.Tanh);

            var teachingSelection = new Selection();
            teachingSelection.Samples.AddRange(new[] 
            { 
                // Side A
                new Sample(new double [] { -3, 0 }, new double [] { 1, -1 }),
                new Sample(new double [] { -3, 1 }, new double [] { 1, -1 }),
                new Sample(new double [] { -3, 2 }, new double [] { 1, -1 }),
                new Sample(new double [] { -3, 3 }, new double [] { 1, -1 }),
                new Sample(new double [] { -2, 3 }, new double [] { 1, -1 }),
                new Sample(new double [] { -1, 3 }, new double [] { 1, -1 }),
                new Sample(new double [] { 0, 3 }, new double [] { 1, -1 }),

                // Side B
                new Sample(new double [] { -1, -3 }, new double [] { -1, 1 }),
                new Sample(new double [] { -1, -2 }, new double [] { -1, 1 }),
                new Sample(new double [] { -1, -1 }, new double [] { -1, 1 }),
                new Sample(new double [] { -1, 0 }, new double [] { -1, 1 }),
                new Sample(new double [] { 0, 1 }, new double [] { -1, 1 }),
                new Sample(new double [] { 1, 1 }, new double [] { -1, 1 }),
                new Sample(new double [] { 2, 1 }, new double [] { -1, 1 }),
                new Sample(new double [] { 3, 1 }, new double [] { -1, 1 }),
                new Sample(new double [] { -0.9, 0.9 }, new double [] { -1, 1 }),
            });

            var controlSelection = new Selection();
            controlSelection.Samples.AddRange( new[] 
            { 
                // Potential errors
                new Sample(new double [] { -1.1, 1.1 }, new double [] { 1, -1 }),
                new Sample(new double [] { -2.5, -1 }, new double [] { -1, 1 }),
                new Sample(new double [] { 1, 2.5 }, new double [] { -1, 1 })
            });

            nn.TrainWithLossType(LossFunctionType.Hinge, 300, teachingSelection, controlSelection);
            nn.CheckWithSelection(teachingSelection);
            nn.CheckWithSelection(controlSelection);
        }
        /// <summary>
        /// The task about a small NN that teaches to convert temperature values
        /// from the Celsius degrees to the Farenheit degrees.
        /// This kind of problems helps to figure out the influence of the weights between
        /// neurons and biases. 
        /// It also cold be convinient to implement the objective functions.
        /// WSe use 1-1 linear NN architecture
        /// </summary>
        private static void CelsiusToFarenheit()
        {
            var nn = NeuralNetworkFactory
                .CreateByFunctionTypeAndLayerSizes(ActivationFunctionType.Identity, new[] { 1, 1 }, 0.001);

            var teachingSelection = new Selection();
            teachingSelection.Samples.AddRange(new[]
            {
                new Sample(new double[] {-40}, new double[] {-40}),
                new Sample(new double[] {-10}, new double[] {14}),
                new Sample(new double[] {0}, new double[] {32}),
                new Sample(new double[] {8}, new double[] {46}),
                new Sample(new double[] {15}, new double[] {59}),
                new Sample(new double[] {22}, new double[] {72}),
                new Sample(new double[] {38}, new double[] {100}),
            });
            nn.TrainWithSelection(teachingSelection, 1000);
            nn.CheckWithSelection(teachingSelection);
            nn.CheckWithInputVector(new double[] { 100 }); // Should be 212
            var weights = nn.OutputNeurons.First().Synapses.Select(s => s.Weight);
            Console.WriteLine($"Weights are: {string.Join(", ", weights)}");
        }

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
        /// Having Biases, we don't need a hidden layer
        /// 
        /// </summary>
        private static void SegmentDigits()
        {
            var nn = NeuralNetworkFactory
                .CreateByFunctionTypeAndLayerSizes(ActivationFunctionType.Logistic, new[] { 7, 10 }, 1);

            var selection = new Selection();
            selection.Samples.AddRange(new[] {
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

            nn.TrainWithSelection(selection, 400);
            nn.CheckWithSelection(selection);
        }
    }
}
