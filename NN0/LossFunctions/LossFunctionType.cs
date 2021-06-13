using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NN0.LossFunctions
{
    public enum LossFunctionType
    {
        /// <summary>
        /// Mean squared error. For regression tasks.
        /// </summary>
        MSE,
        /// <summary>
        /// Mean absolute error. For regression tasks.
        /// </summary>
        MAE,
        /// <summary>
        /// Mean absolute percentage error. For regression tasks.
        /// </summary>
        MAPE,
        /// <summary>
        /// Mean squared logarithmic error. For regression tasks.
        /// </summary>
        MSLE,
        /// <summary>
        /// Hinge. For recognition tasks.
        /// </summary>
        Hinge,
        /// <summary>
        /// Hinge. For recognition tasks.
        /// </summary>
        SquaredHinge,
        /// <summary>
        /// Binary crossentropy. For recognition between two types.
        /// </summary>
        BinaryCrossentropy,
        /// <summary>
        /// Categotical crossEntropy. For recognition by multiple categories.
        /// </summary>
        CathegoicalCrossEntropy,
        /// <summary>
        /// logcosh. For text recognition tasks.
        /// </summary>
        Logcosh
    }
}
