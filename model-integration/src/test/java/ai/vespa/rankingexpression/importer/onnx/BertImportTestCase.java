// Copyright 2019 Oath Inc. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package ai.vespa.rankingexpression.importer.onnx;

import ai.vespa.rankingexpression.importer.ImportedModel;
import com.yahoo.io.IOUtils;
import com.yahoo.searchlib.rankingexpression.RankingExpression;
import com.yahoo.searchlib.rankingexpression.Reference;
import com.yahoo.searchlib.rankingexpression.evaluation.Context;
import com.yahoo.searchlib.rankingexpression.evaluation.ContextIndex;
import com.yahoo.searchlib.rankingexpression.evaluation.DoubleValue;
import com.yahoo.searchlib.rankingexpression.evaluation.MapContext;
import com.yahoo.searchlib.rankingexpression.evaluation.TensorValue;
import com.yahoo.searchlib.rankingexpression.evaluation.Value;
import com.yahoo.searchlib.rankingexpression.parser.ParseException;
import com.yahoo.searchlib.rankingexpression.rule.CompositeNode;
import com.yahoo.searchlib.rankingexpression.rule.ExpressionNode;
import com.yahoo.searchlib.rankingexpression.rule.ReferenceNode;
import com.yahoo.searchlib.rankingexpression.rule.TensorFunctionNode;
import com.yahoo.tensor.Tensor;
import com.yahoo.tensor.TensorType;
import com.yahoo.tensor.functions.Generate;
import com.yahoo.tensor.functions.ScalarFunction;
import com.yahoo.tensor.functions.Slice;
import org.junit.Ignore;
import org.junit.Test;
import org.tensorflow.op.core.Rank;

import java.io.BufferedReader;
import java.io.IOException;
import java.sql.Ref;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import static org.junit.Assert.assertEquals;

/**
 * @author lesters
 */
public class BertImportTestCase extends TestableModel {

    @Test
    public void test() throws Exception {
        String filename = "/Users/lesters/github/onnx-models/text/machine_comprehension/bert-squad/java.txt";
        List<String> lines = IOUtils.getLines(filename);
        Tensor tensor = Tensor.from(lines.get(0));
        TestableModelContext context = new TestableModelContext();
        context.put("test", new TensorValue(tensor));

        // Tensor: tensor(d1[1],d2[256],d4[12],d5[64])

        String expr = "tensor(d0[256],d1[768])" +
                "((test{" +
                    "d1:(floor(0.0)), " +
                    "d2:(floor(((768.0 * d0 + d1) % 196608) / 768.0)), " +
                    "d4:(floor(((768.0 * d0 + d1) % 768.0) / 64.0)), " +
                    "d5:(floor((768.0 * d0 + d1) % 64.0))" +
                "}))";
        Tensor result = new RankingExpression(expr).evaluate(context).asTensor();

        assertEquals(result.sum(), -6074.247);
    }

    @Ignore
    @Test
    public void testBertImport() {
        ImportedModel model = new OnnxImporter().importModel("test", "/Users/lesters/github/onnx-models/text/machine_comprehension/bert-squad/bertsquad8_modified.onnx");
//        ImportedModel model = new OnnxImporter().importModel("test", "src/test/models/onnx/bert/bertsquad8_modified.onnx");
//        ImportedModel model = new OnnxImporter().importModel("test", "src/test/models/onnx/bert/bertsquad10.onnx");
//        assertEquals(0, model.signature("default").skippedOutputs().size());
//        Tensor onnxResult = evaluateVespa(model, "output", model.inputs());
//        assertEquals(Tensor.from("tensor(d0[1],d1[2]):[[0.28258783057229725, -0.0685615853647904]]"), onnxResult);

        String filename = "/Users/lesters/github/onnx-models/text/machine_comprehension/bert-squad/context.vespa";

        // bert/encoder/layer_0/attention/self/mul_2
        assert null != model.largeConstants().get("test_bert_encoder_layer_0_attention_self_Reshape_3__294");

        TestableModelContext context;
        if (true) {
            // inputs
            Tensor unique_ids_raw_output__9 = Tensor.from("tensor(d0[1]):[1]");
            Tensor input_ids = Tensor.from("tensor(d0[1],d1[256]):[101,2073,2003,1996,5661,10549,2000,2175,1029,102,1999,2049,2220,2086,1010,1996,2047,4680,2415,3478,2000,3113,5270,1998,6599,10908,1012,1031,2260,1033,2011,2526,1010,2116,13773,3028,5661,2020,10549,1996,2172,3469,9587,9363,2638,2415,1999,2624,3799,2058,1996,2624,4560,4680,2415,2349,2000,1996,3732,1005,1055,3132,2686,1012,1037,10428,5468,2000,5446,2019,4935,3081,1037,3309,4171,3478,2000,3362,1996,3223,2048,1011,12263,3484,2000,3413,1012,1999,2238,2384,1010,2136,2624,4560,2328,1996,2148,2534,1010,1037,1002,1020,1012,6255,2454,1010,2630,1998,2317,9311,1010,5815,3770,1010,2199,2675,2519,1006,1021,1010,4278,25525,1007,1997,8327,2686,102,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]");
            Tensor input_mask = Tensor.from("tensor(d0[1],d1[256]):[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]");
            Tensor segment_ids = Tensor.from("tensor(d0[1],d1[256]):[0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]");

            context = contextFrom(model);
            context.put("unique_ids_raw_output___9", new TensorValue(unique_ids_raw_output__9));
            context.put("input_ids", new TensorValue(input_ids));
            context.put("input_mask", new TensorValue(input_mask));
            context.put("segment_ids", new TensorValue(segment_ids));

            context.write(filename);
        } else {
            context = TestableModelContext.read(filename);
        }

        // expected outputs from onnxruntime
        Tensor unique_ids = Tensor.from("tensor(d0[1]):[1]");
        Tensor unstack_0 = Tensor.from("tensor(d0[1],d1[256]):[-7.169589,-8.165145,-8.795558,-8.276284,-8.408593,-8.313643,-8.421538,-8.771402,-8.71111,-8.014886,-6.4415646,-7.5764513,-7.7209125,-8.689668,-8.05441,-4.357495,-4.082243,-2.4557219,-6.421309,-7.8627315,-8.612887,-8.122109,-8.072487,-8.678347,-8.467162,-8.818881,-8.143695,-6.412044,-7.765201,-8.125683,-2.5739796,-4.2929254,-7.8812947,-2.4893682,-1.5166948,-6.2354407,-4.039099,-5.6837378,-0.41342238,3.0958412,1.5454307,-0.89450985,6.0985346,-4.108738,-4.67186,-2.6797965,-0.65007347,4.2300944,-2.9132476,-4.853151,1.1584995,4.041984,-3.5257776,-2.3050616,-4.363427,-7.1510825,-8.426602,-6.6682553,-7.027374,-8.076435,-8.3017435,-6.9958987,-7.243815,-7.1347113,-7.5506253,-7.771371,-8.606251,-7.472072,-7.902196,-7.563202,-7.330995,-7.5767503,-7.8097973,-6.645113,-8.927777,-8.438513,-8.708496,-8.474434,-8.231956,-8.635139,-7.8764973,-8.80273,-9.103729,-9.07057,-8.610826,-9.084642,-8.795743,-7.2711506,-7.733648,-8.708181,-8.020964,-7.1652384,-7.9469404,-9.461184,-8.624146,-7.2252526,-6.4015207,-9.220176,-9.195709,-8.228707,-7.9646325,-8.685807,-8.980191,-8.858017,-9.290145,-8.921865,-7.656322,-8.872562,-8.898288,-8.683226,-9.219653,-8.371141,-7.130355,-8.930712,-9.05438,-8.771264,-9.621703,-8.550959,-7.327657,-9.138217,-9.377564,-9.111144,-9.653343,-8.726485,-8.215803,-9.300696,-8.044907,-8.641199,-8.641449,-8.640882,-8.641207,-8.648375,-8.645785,-8.639973,-8.650788,-8.660226,-8.6503525,-8.6601925,-8.647732,-8.652576,-8.665123,-8.653585,-8.653888,-8.661093,-8.661934,-8.6514845,-8.662573,-8.671499,-8.661195,-8.667901,-8.666959,-8.659721,-8.673244,-8.678537,-8.66441,-8.651034,-8.660175,-8.659063,-8.657169,-8.6603565,-8.6569,-8.649067,-8.651927,-8.6421995,-8.649052,-8.6478615,-8.6426935,-8.646153,-8.646865,-8.636821,-8.643324,-8.645994,-8.639597,-8.647679,-8.655649,-8.6609745,-8.654906,-8.6613455,-8.656511,-8.663024,-8.675192,-8.663131,-8.665018,-8.652522,-8.661668,-8.66894,-8.670112,-8.67217,-8.657303,-8.651893,-8.652592,-8.650168,-8.640702,-8.636455,-8.647628,-8.638621,-8.648,-8.656844,-8.649821,-8.657603,-8.648884,-8.661986,-8.663507,-8.652322,-8.662775,-8.664504,-8.662872,-8.668943,-8.6559105,-8.655738,-8.671845,-8.6666,-8.659552,-8.679308,-8.659756,-8.664594,-8.6688175,-8.666396,-8.673796,-8.65924,-8.664916,-8.6703005,-8.6611395,-8.660061,-8.660967,-8.672797,-8.66394,-8.657039,-8.671023,-8.663469,-8.659371,-8.6713705,-8.659359,-8.649764,-8.6620035,-8.656843,-8.654225,-8.661666,-8.647326,-8.652874,-8.650523,-8.644273,-8.649993,-8.65307,-8.645219,-8.6537075,-8.655814,-8.654312,-8.658724,-8.666763,-8.654713,-8.662302,-8.672376,-8.661079,-8.659652,-8.661736]");
        Tensor unstack_1 = Tensor.from("tensor(d0[1],d1[256]):[-5.1743593,-8.167716,-8.096918,-8.610186,-8.627197,-8.518608,-8.413071,-8.04796,-8.405228,-5.775467,-8.891069,-8.499419,-8.482899,-7.4575906,-5.1060586,-9.029796,-7.9796743,-7.411322,-0.62632525,-8.209348,-8.202109,-8.436105,-8.226212,-8.245562,-7.7150273,-5.4672513,-6.134469,-8.531252,-7.390566,-6.717802,-7.9110403,-5.084878,-5.02966,-7.6901536,-7.6643076,-0.42670453,-4.2289968,-6.957412,-5.192218,-6.1616683,-6.4489427,-3.5914042,-3.7853065,-6.857571,-2.3781726,6.1620126,-3.007885,-4.688912,6.258016,-5.2202945,-6.7945094,-5.1450105,0.7468612,-4.919924,5.489712,-7.307814,-7.952,-9.152897,-6.4863043,-8.328119,-7.9448185,-8.395245,-3.6581624,-0.8252581,-8.731679,-8.624653,-7.61354,-8.755644,-8.341698,-8.758186,-5.954141,-8.560192,-8.833243,-7.6137505,-5.96118,-8.43961,-8.188338,-8.373185,-8.683964,-8.246368,-8.824446,-8.05728,-7.623751,-7.56998,-8.277908,-6.8986,-6.8709283,-9.279125,-8.84588,-7.6791453,-5.2976,-9.191589,-8.797903,-6.440836,-8.179676,-9.236156,-8.972708,-5.8724217,-6.928253,-8.685118,-8.84946,-8.293621,-7.8572874,-8.053903,-7.398021,-7.549705,-9.004784,-8.060446,-7.950672,-7.2188964,-6.497633,-8.454956,-9.045556,-7.8463507,-7.771165,-8.067679,-5.9176393,-8.09684,-8.5619955,-7.5696144,-7.100621,-7.0136676,-6.464568,-8.108538,-8.516457,-5.488856,-5.853514,-8.457255,-8.457188,-8.454844,-8.448273,-8.447864,-8.447953,-8.445874,-8.444903,-8.442595,-8.448225,-8.443758,-8.451776,-8.447646,-8.440473,-8.447313,-8.44705,-8.443977,-8.442994,-8.449743,-8.441086,-8.433916,-8.43898,-8.435363,-8.434594,-8.431777,-8.433416,-8.433545,-8.442987,-8.453411,-8.450068,-8.4503565,-8.451651,-8.450909,-8.454222,-8.456041,-8.452284,-8.449699,-8.454986,-8.455096,-8.459543,-8.458114,-8.458371,-8.4632635,-8.458183,-8.457299,-8.458008,-8.452067,-8.444335,-8.442348,-8.445211,-8.441855,-8.443939,-8.441303,-8.436119,-8.442878,-8.439337,-8.446676,-8.441184,-8.438475,-8.440033,-8.4386015,-8.447922,-8.455316,-8.452563,-8.454967,-8.459164,-8.460839,-8.453004,-8.451543,-8.446279,-8.441412,-8.448481,-8.446184,-8.448539,-8.445241,-8.444487,-8.450539,-8.446448,-8.446319,-8.447268,-8.440758,-8.448286,-8.447366,-8.437631,-8.441085,-8.444475,-8.431786,-8.441355,-8.436929,-8.432141,-8.436456,-8.435032,-8.445299,-8.442143,-8.438964,-8.445743,-8.445099,-8.444958,-8.438029,-8.439503,-8.446831,-8.43919,-8.442334,-8.446472,-8.442076,-8.449043,-8.451941,-8.449556,-8.454564,-8.455859,-8.452123,-8.461076,-8.45802,-8.456931,-8.458485,-8.45496,-8.4508295,-8.453123,-8.451649,-8.451098,-8.450148,-8.446929,-8.44253,-8.44839,-8.444667,-8.437894,-8.444409,-8.444666,-8.441956]");


//        model.functions().forEach((k, v) -> {
//            evaluateFunction(context, model, k, "");
//        });

//        RankingExpression e = model.expressions().get("unique_ids_graph_outputs_Identity__10");
//        evaluateFunctionDependencies(context, model, e.getRoot(), "");
//        Tensor result = e.evaluate(context).asTensor();
//        assertEquals(result, unique_ids);

        RankingExpression e = model.expressions().get("bert/encoder/layer_0/output/LayerNorm/batchnorm/add_1");

        evaluateFunctionDependencies(context, model, e.getRoot(), "");
        context.write(filename);
        Tensor result = e.evaluate(context).asTensor();
        double sum = result.sum().asDouble();
        System.out.println(sum);

        Tensor matmul1 = model.expressions().get("bert/encoder/layer_0/attention/self/MatMul_1").evaluate(context).asTensor();

        Tensor transpose = model.expressions().get("bert/encoder/layer_0/attention/self/transpose_3").evaluate(context).asTensor();
        String cast = model.largeConstants().get("test_bert_encoder_layer_0_attention_self_Reshape_3__294");
        Tensor reshape = model.expressions().get("bert/encoder/layer_0/attention/self/Reshape_3").evaluate(context).asTensor();
        Tensor matmul = model.expressions().get("bert/encoder/layer_0/attention/output/dense/MatMul").evaluate(context).asTensor();
        Tensor add = model.expressions().get("bert/encoder/layer_0/attention/output/dense/BiasAdd").evaluate(context).asTensor();

        Tensor add1 = model.expressions().get("bert/encoder/layer_0/attention/output/add").evaluate(context).asTensor();
        Tensor add2 = model.expressions().get("bert/encoder/layer_0/attention/output/LayerNorm/batchnorm/add_1").evaluate(context).asTensor();
        Tensor add3 = model.expressions().get("bert/encoder/layer_0/output/add").evaluate(context).asTensor();
        Tensor add4 = model.expressions().get("bert/encoder/layer_0/output/LayerNorm/batchnorm/add_1").evaluate(context).asTensor();

        assertEquals(result, unique_ids);

//        Tensor result = model.expressions().get("unique_ids_graph_outputs_Identity__10").evaluate(context).asTensor();
//        assertEquals(result, unique_ids);

//        result = model.expressions().get("unstack_graph_outputs_Identity__7").evaluate(context).asTensor();  // or map from signature outputs
//        assertEquals(result, unstack_0);

        // en feil her i outputs: har bare en: unstack, men vi må ha to: unstack:0 og unstack:1

    }


    private void evaluateFunction(Context context, ImportedModel model, String functionName, String in) {
        if (!context.names().contains(functionName)) {
            RankingExpression e = RankingExpression.from(model.functions().get(functionName));
            System.out.println(in + "Looking for dependencies of function " + functionName + ": " + e.toString());
            evaluateFunctionDependencies(context, model, e.getRoot(), in);
            System.out.println(in + "Evaluating function " + functionName + ": " + e.toString());
            long start = System.currentTimeMillis();
            Tensor result = e.evaluate(context).asTensor();
            context.put(functionName, new TensorValue(result));
            long end = System.currentTimeMillis();
            System.out.println(in + "[" + (end - start) + "] completed " + functionName + " (" + result.type() + "), context is: " + context.names().size() + " " + contextSize(context));
        } else {
            System.out.println(in + "Function " + functionName + " already evaluated...");
        }
    }

    private long contextSize(Context context) {
        long size = 0;
        for (String name : context.names()) {
            Tensor val = context.getTensor(name);
            if (val != null) size += val.size();
        }
        return size;
    }

    private void evaluateFunctionDependencies(Context context, ImportedModel model, ExpressionNode node, String in) {
        if (node instanceof ReferenceNode) {
            String name = node.toString();
            ReferenceNode ref = (ReferenceNode) node;
            if (ref.getName().equals("constant")) {
                String constant = ref.getArguments().expressions().get(0).toString();
                if (!context.names().contains(constant)) {
                    String value = null;
                    if (model.smallConstants().containsKey(constant)) {
                        value = model.smallConstants().get(constant);
                    }
                    if (model.largeConstants().containsKey(constant)) {
                        value = model.largeConstants().get(constant);
                    }
                    if (value != null) {
                        System.out.println(in + "Adding constant: " + name);
                        long start = System.currentTimeMillis();
                        Tensor val = Tensor.from(value);
                        context.put(name, new TensorValue(val));
                        long end = System.currentTimeMillis();
                        System.out.println(in + "Added constant: " + name + " (" + val.type() + ") in [" + (end - start) + "]");
                    }
                }
            }
            if (model.functions().containsKey(name)) {
                evaluateFunction(context, model, name, in + " ");
            }
        }
        else if (node instanceof CompositeNode) {
            if (node instanceof TensorFunctionNode && ((TensorFunctionNode)node).function() instanceof Generate) {
                Generate generate = (Generate) ((TensorFunctionNode)node).function();
                TensorFunctionNode.ExpressionScalarFunction func = (TensorFunctionNode.ExpressionScalarFunction) generate.getBoundGenerator();
                if (func != null) {
                    ExpressionNode bound = func.getExpression();
                    if (bound.toString().contains("imported_ml_")) {
                        System.out.println(in + "Found expression inside generator: " + bound.toString());
                        evaluateFunctionDependencies(context, model, bound, in);
                    }
                }
            }
            else if (node instanceof TensorFunctionNode && ((TensorFunctionNode)node).function() instanceof Slice) {
                Slice<Reference> slice = (Slice<Reference>) ((TensorFunctionNode)node).function();
                for (Slice.DimensionValue<Reference> value : slice.getSubspaceAddress()) {
                    TensorFunctionNode.ExpressionScalarFunction func = (TensorFunctionNode.ExpressionScalarFunction) value.index().orElse(null);
                    if (func != null) {
                        ExpressionNode bound = func.getExpression();
                        if (bound.toString().contains("imported_ml_")) {
                            System.out.println(in + "Found expression inside slice: " + bound.toString());
                            evaluateFunctionDependencies(context, model, bound, in);
                        }
                    }
                }
            }
            for (ExpressionNode child : ((CompositeNode)node).children()) {
                evaluateFunctionDependencies(context, model, child, in);
            }
        }
    }

    static TestableModelContext contextFrom(ImportedModel result) {
        TestableModelContext context = new TestableModelContext();
        if (result != null) {
            result.largeConstants().forEach((name, tensor) -> context.put("constant(" + name + ")", new TensorValue(Tensor.from(tensor))));
            result.smallConstants().forEach((name, tensor) -> context.put("constant(" + name + ")", new TensorValue(Tensor.from(tensor))));
        }
        return context;
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

        public void write(String filename) {
            try {
                for (Map.Entry<String, Value> entry: bindings().entrySet()) {
                String line = entry.getKey() + "\t" + entry.getValue().asTensor() + "\n";
                IOUtils.writeFile(filename, line, true);
            }
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        }

        public static TestableModelContext read(String filename) {
            System.out.println("Reading content from " + filename);
            TestableModelContext context = new TestableModelContext();
            try (BufferedReader reader = IOUtils.createReader(filename)) {
                String line;
                while (null != (line = reader.readLine())) {
                    String[] strings = line.trim().split("\t");
                    String name = strings[0];
                    Tensor tensor = Tensor.from(strings[1]);
                    context.put(name, new TensorValue(tensor));
                }
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
            System.out.println("Done reading context");
            return context;
        }
    }

}
