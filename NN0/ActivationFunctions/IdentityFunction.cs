using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NN0.ActivationFunctions
{
    public class IdentityFunction : ILayerIndependentFunction
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
