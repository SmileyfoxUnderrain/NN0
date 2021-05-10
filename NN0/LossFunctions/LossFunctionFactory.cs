using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NN0.LossFunctions
{
    public class LossFunctionFactory
    {
        public static ILossFunction GetByType(LossFunctionType type)
        {
            if (type == LossFunctionType.MSE)
                return new MeanSquaredErrorFunction();
            if (type == LossFunctionType.MAE)
                return new MeanAbsoluteErrorFunction();
            if (type == LossFunctionType.MAPE)
                return new MeanAbsolutePercentageErrorFunction();
            if (type == LossFunctionType.MSLE)
                return new MeanSquaredLogarithmicErrorFunction();
            if (type == LossFunctionType.Hinge)
                return new HingeFunction();
            if (type == LossFunctionType.SquaredHinge)
                return new SquaredHingeFunction();

            throw new NotImplementedException();
        }
    }
}
