// Copyright 2018 Yahoo Holdings. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package ai.vespa.rankingexpression.importer.operations;

import ai.vespa.rankingexpression.importer.DimensionRenamer;
import ai.vespa.rankingexpression.importer.OrderedTensorType;
import com.yahoo.tensor.TensorType;
import com.yahoo.tensor.functions.Join;
import com.yahoo.tensor.functions.Reduce;
import com.yahoo.tensor.functions.ScalarFunctions;
import com.yahoo.tensor.functions.TensorFunction;
import com.yahoo.text.ExpressionFormatter;

import java.util.List;
import java.util.Optional;

public class MatMul extends IntermediateOperation {

    public MatMul(String modelName, String nodeName, List<IntermediateOperation> inputs) {
        super(modelName, nodeName, inputs);
    }

    @Override
    protected OrderedTensorType lazyGetType() {
        if ( ! allInputTypesPresent(2)) return null;

        OrderedTensorType aType = inputs.get(0).type().get();
        OrderedTensorType bType = inputs.get(1).type().get();

        // add some more checks here
        if (aType.type().rank() < 1 || bType.type().rank() < 1)
            throw new IllegalArgumentException("Tensors in matmul must have rank of at least 1");

        OrderedTensorType.Builder typeBuilder = new OrderedTensorType.Builder(resultValueType());
        OrderedTensorType largestRankType = aType.rank() >= bType.rank() ? aType : bType;
        for (int i = 0; i < largestRankType.rank() - 2; ++i) {
            typeBuilder.add(largestRankType.dimensions().get(i));
        }
        if (aType.rank() >= 2) {
            typeBuilder.add(aType.dimensions().get(aType.rank() - 2));
        }
        if (bType.rank() >= 2) {
            typeBuilder.add(bType.dimensions().get(bType.rank() - 1));
        }
        return typeBuilder.build();
    }

    @Override
    protected TensorFunction lazyGetFunction() {
        if ( ! allInputTypesPresent(2)) return null;
        if ( ! allInputFunctionsPresent(2)) return null;

        OrderedTensorType aType = inputs.get(0).type().get();
        Optional<TensorFunction> aFunction = inputs.get(0).function();
        Optional<TensorFunction> bFunction = inputs.get(1).function();

        // only change to this is for dimensions with size 1 - check in getType

        return new com.yahoo.tensor.functions.Reduce(new Join(aFunction.get(), bFunction.get(), ScalarFunctions.multiply()),
                Reduce.Aggregator.sum,
                aType.dimensions().get(aType.rank() - 1).name());
    }

    @Override
    public void addDimensionNameConstraints(DimensionRenamer renamer) {
        if ( ! allInputTypesPresent(2)) return;

        /*
         * A: a1, a2, a3, a4
         * B: b1, b2, b3, b4
         *
         * a4 == b3
         * a3 < b4
         * a3 < a4
         * b4 < b3
         *
         * a1 == b1 -> men også størrelsesmessig.
         * a2 == b2
         * etc
         */

        OrderedTensorType typeA = inputs.get(0).type().get();
        OrderedTensorType typeB = inputs.get(1).type().get();

        String lastDimA = typeA.dimensions().get(typeA.rank()-1).name();
        String lastDimB = typeB.dimensions().get(typeB.rank()-1).name();
        String secondLastDimA = typeA.dimensions().get(Math.max(0,typeA.rank()-2)).name();
        String secondLastDimB = typeB.dimensions().get(Math.max(0,typeB.rank()-2)).name();

        // The last dimension of A should have the same name as the second-to-last dimension of B
        renamer.addConstraint(lastDimA, secondLastDimB, DimensionRenamer.Constraint.equal(false), this);

        // For efficiency, the dimensions to join over should be innermost - soft constraint
        if (typeA.rank() >= 2) {
            renamer.addConstraint(secondLastDimA, lastDimA, DimensionRenamer.Constraint.lessThan(true), this);
        }
        if (typeB.rank() >= 2) {
            renamer.addConstraint(secondLastDimB, lastDimB, DimensionRenamer.Constraint.greaterThan(true), this);
        }

        // The second-to-last dimension of a should have a different name than the last dimension of b
        if (typeA.rank() >= 2 && typeB.rank() >= 2) {
            renamer.addConstraint(secondLastDimA, lastDimB, DimensionRenamer.Constraint.lessThan(false), this);
        }

        // a1 < a2 < a3 < a4
        OrderedTensorType largestRankType = typeA.rank() >= typeB.rank() ? typeA : typeB;
        for (int i = 0; i < largestRankType.rank() - 2; ++i) {
            String iDim = largestRankType.dimensionNames().get(i);
            for (int j = i+1; j < largestRankType.rank() - 2; ++j) {
                String jDim = largestRankType.dimensionNames().get(j);
                renamer.addConstraint(iDim, jDim, DimensionRenamer.Constraint.lessThan(true), this);
            }
        }

        // TODO: handle non similar sizes

        // a1 == b1 etc
        if (typeA.rank() == typeB.rank()) {
            for (int i = 0; i < typeA.rank() - 2; ++i) {
                renamer.addConstraint(typeA.dimensionNames().get(i), typeB.dimensionNames().get(i), DimensionRenamer.Constraint.equal(false), this);
            }
        }




        // So, what about the other dimensions?
//        if (aDimensions.size() > 2) {
//            for (int i = 1; i < aDimensions.size(); ++i) {
//                renamer.addConstraint(aDimensions.get(0).name(), aDimensions.get(i).name(), DimensionRenamer.Constraint.notEqual(false), this);
//            }
//            for (int i = 0; i < bDimensions.size(); ++i) {
//                renamer.addConstraint(aDimensions.get(0).name(), bDimensions.get(i).name(), DimensionRenamer.Constraint.notEqual(false), this);
//            }
//        }

    }

//    private void assertTwoDimensions(List<TensorType.Dimension> dimensions, IntermediateOperation supplier, String inputDescription) {
//        if (dimensions.size() >= 2) return;
//        throw new IllegalArgumentException("Expected 2 dimensions in the " + inputDescription + " to " + this +
//                                           " but got just " + dimensions + " from\n" +
//                                           ExpressionFormatter.inTwoColumnMode(70, 50).format(supplier.toFullString()));
//    }

    @Override
    public MatMul withInputs(List<IntermediateOperation> inputs) {
        return new MatMul(modelName(), name(), inputs);
    }

    @Override
    public String operationName() { return "MatMul"; }

}
