using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NN0.ActivationFunctions
{
    public class EluFunction : ILayerIndependentFunction
    {
        
        public EluFunction(double alpha)
        {
            Alpha = alpha;
        }
        public double Alpha { get; set; }
        public double Function(double x)
        {
            var ret = x > 0
                ? x
                : Alpha * (Math.Pow(Math.E, x) - 1);
            return ret;
        }
        public double Derivative(double x)
        {
            
            if (x == 0 && Alpha == 0)
                return 1;
            if (x > 0)
                return 1;

            var ret = Alpha * Math.Pow(Math.E, x);
            return ret;
        }
    }
}
