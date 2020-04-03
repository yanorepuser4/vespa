// Copyright 2018 Yahoo Holdings. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package ai.vespa.rankingexpression.importer.tensorflow;

import ai.vespa.rankingexpression.importer.ImportedModel;
import ai.vespa.rankingexpression.importer.configmodelview.ImportedMlFunction;
import ai.vespa.rankingexpression.importer.onnx.OnnxImporter;
import com.yahoo.collections.Pair;
import com.yahoo.searchlib.rankingexpression.RankingExpression;
import com.yahoo.searchlib.rankingexpression.evaluation.Context;
import com.yahoo.searchlib.rankingexpression.evaluation.ContextIndex;
import com.yahoo.searchlib.rankingexpression.evaluation.ExpressionOptimizer;
import com.yahoo.searchlib.rankingexpression.evaluation.MapContext;
import com.yahoo.searchlib.rankingexpression.evaluation.TensorValue;
import com.yahoo.searchlib.rankingexpression.rule.CompositeNode;
import com.yahoo.searchlib.rankingexpression.rule.ExpressionNode;
import com.yahoo.searchlib.rankingexpression.rule.ReferenceNode;
import com.yahoo.system.ProcessExecuter;
import com.yahoo.tensor.Tensor;
import com.yahoo.tensor.TensorType;
import org.junit.Assert;
import org.junit.Ignore;
import org.junit.Test;
import org.tensorflow.SavedModelBundle;
import org.tensorflow.Session;

import java.io.IOException;
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

public class LesterTensorflowImportTestCase {

    @Test
    @Ignore
    public void testPyTorchExport() {
        ImportedModel model = new OnnxImporter().importModel("test", "src/test/models/pytorch/test.onnx");
        Tensor onnxResult = evaluateVespa(model, "output", model.inputs());
        assertEquals(Tensor.from("tensor(d0[1],d1[2]):[[0.2134095202835272, -0.08556838456161658]]"), onnxResult);
    }

    @Test
    @Ignore
    public void testBERT() {
        ImportedModel model = new OnnxImporter().importModel("test", "src/test/models/onnx/bert/bertsquad10.onnx");
    }

    private Tensor evaluateVespa(ImportedModel model, String operationName, Map<String, TensorType> inputs) {
        Context context = contextFrom(model);
        for (Map.Entry<String, TensorType> entry : inputs.entrySet()) {
            Tensor argument = vespaInputArgument(1, entry.getValue().dimensions().get(1).size().get().intValue());
            context.put(entry.getKey(), new TensorValue(argument));
        }
        model.functions().forEach((k, v) -> evaluateFunction(context, model, k));
        RankingExpression expression = model.expressions().get(operationName);
        ExpressionOptimizer optimizer = new ExpressionOptimizer();
        optimizer.optimize(expression, (ContextIndex)context);
        return expression.evaluate(context).asTensor();
    }

    @Test
    @Ignore
    public void testModelImport() {

        // Må endre til tf 2.0 i java!

        String modelDir = "src/test/models/tensorflow/tf2/saved_model/";
        // output function
        String operationName = "out";

        // Import TF
        SavedModelBundle tensorFlowModel = SavedModelBundle.load(modelDir, "serve");
        ImportedModel model = new TensorFlowImporter().importModel("test", modelDir, tensorFlowModel);
        ImportedModel.Signature signature = model.signature("serving_default");
        assertEquals("Should have no skipped outputs", 0, model.signature("serving_default").skippedOutputs().size());
        ImportedMlFunction output = signature.outputFunction("output", operationName);
        assertNotNull(output);

        // Test TF
        Session.Runner runner = tensorFlowModel.session().runner();
        runner.feed("x", tensorFlowFloatInputArgument(1, 4));
        List<org.tensorflow.Tensor<?>> results = runner.fetch(operationName).run();
        assertEquals(1, results.size());
        Tensor tfResult = TensorConverter.toVespaTensor(results.get(0));

        // Test Vespa
        Context context = contextFrom(model);
        context.put("x", new TensorValue(vespaInputArgument(1, 4)));
        model.functions().forEach((k, v) -> evaluateFunction(context, model, k));
        RankingExpression expression = model.expressions().get(operationName);
        ExpressionOptimizer optimizer = new ExpressionOptimizer();
        optimizer.optimize(expression, (ContextIndex)context);
        Tensor vespaResult = expression.evaluate(context).asTensor();

        // Equal result?
        System.out.println(tfResult);
        System.out.println(vespaResult);
        assertEquals(tfResult, vespaResult);
    }

    private org.tensorflow.Tensor<?> tensorFlowFloatInputArgument(int d0Size, int d1Size) {
        FloatBuffer fb1 = FloatBuffer.allocate(d0Size * d1Size);
        int i = 0;
        for (int d0 = 0; d0 < d0Size; d0++)
            for (int d1 = 0; d1 < d1Size; ++d1)
                fb1.put(i++, (float)(d1 * 1.0 / d1Size));
        return org.tensorflow.Tensor.create(new long[]{ d0Size, d1Size }, fb1);
    }

    private Tensor vespaInputArgument(int d0Size, int d1Size) {
        Tensor.Builder b = Tensor.Builder.of(new TensorType.Builder().indexed("d0", d0Size).indexed("d1", d1Size).build());
        for (int d0 = 0; d0 < d0Size; d0++)
            for (int d1 = 0; d1 < d1Size; d1++)
                b.cell(d1 * 1.0 / d1Size, d0, d1);
        return b.build();
    }

    static Context contextFrom(ImportedModel result) {
        TestableModelContext context = new TestableModelContext();
        result.largeConstants().forEach((name, tensor) -> context.put("constant(" + name + ")", new TensorValue(Tensor.from(tensor))));
        result.smallConstants().forEach((name, tensor) -> context.put("constant(" + name + ")", new TensorValue(Tensor.from(tensor))));
        return context;
    }

    private void evaluateFunction(Context context, ImportedModel model, String functionName) {
        if (!context.names().contains(functionName)) {
            RankingExpression e = RankingExpression.from(model.functions().get(functionName));
            evaluateFunctionDependencies(context, model, e.getRoot());
            context.put(functionName, new TensorValue(e.evaluate(context).asTensor()));
        }
    }

    private void evaluateFunctionDependencies(Context context, ImportedModel model, ExpressionNode node) {
        if (node instanceof ReferenceNode) {
            String name = node.toString();
            if (model.functions().containsKey(name)) {
                evaluateFunction(context, model, name);
            }
        }
        else if (node instanceof CompositeNode) {
            for (ExpressionNode child : ((CompositeNode)node).children()) {
                evaluateFunctionDependencies(context, model, child);
            }
        }
    }

    private static class TestableModelContext extends MapContext implements ContextIndex {
        @Override
        public int size() {
            return bindings().size();
        }
        @Override
        public int getIndex(String name) {
            throw new UnsupportedOperationException(this + " does not support index lookup by name");
        }
    }

}
