using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NN0.Functions
{
    public class IdentityFunction : IActivationFunction
    {
        public double Function(double x)
        {
            return x;
        }
        public double Derivative(double x)
        {
            return 1;
        }
    }
}
