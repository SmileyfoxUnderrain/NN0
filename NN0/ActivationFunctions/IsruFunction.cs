using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NN0.ActivationFunctions
{
    public class IsruFunction : ILayerIndependentFunction
    {
        
        public double Alpha { get; set; }

        public IsruFunction(double alpha)
        {
            Alpha = alpha;
        }

        public double Function(double x)
        {
            var ret = x / Math.Sqrt(1 + Alpha * Math.Pow(x, 2));
            return ret;
        }
        public double Derivative(double x)
        {
            var ret = Math.Pow(1 / Math.Sqrt(1 + Alpha * Math.Pow(x, 2)), 3);
            return ret;
        }
    }
}
