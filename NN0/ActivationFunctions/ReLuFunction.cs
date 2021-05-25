using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NN0.ActivationFunctions
{
    public class ReLuFunction : ILayerIndependentFunction
    {
        public double Function(double x)
        {
            return x > 0 ? x : 0;
        }
        public double Derivative(double x)
        {
            // But ithe derivative of RelU function is undefined in 0
            return x > 0 ? 1 : 0;
        }
    }
}
