using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NN0.Functions
{
    public class LeakyReLuFunction : IActivationFunction
    {
        private const double LEAKY_COEFFICIENT = 0.01;
        public double Function(double x)
        {
            return x < 0
                ? LEAKY_COEFFICIENT * x
                : x;

        }
        public double Derivative(double x)
        {
            return x < 0
                ? LEAKY_COEFFICIENT
                : 1;
        }
    }
}
