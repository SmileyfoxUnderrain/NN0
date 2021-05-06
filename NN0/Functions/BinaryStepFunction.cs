using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NN0.Functions
{
    class BinaryStepFunction : IActivationFunction
    {
        public double Function(double x)
        {
            return x < 0 ? 0 : 1;
        }
        public double Derivative(double x)
        {
            // If x == 0 the value is undefined
            return 0;
        }
    }
}
