// Copyright 2018 Yahoo Holdings. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package com.yahoo.searchdefinition.processing;

import ai.vespa.rankingexpression.importer.configmodelview.ImportedMlModels;
import ai.vespa.rankingexpression.importer.configmodelview.MlModelImporter;
import ai.vespa.rankingexpression.importer.lightgbm.LightGBMImporter;
import ai.vespa.rankingexpression.importer.onnx.OnnxImporter;
import ai.vespa.rankingexpression.importer.tensorflow.TensorFlowImporter;
import ai.vespa.rankingexpression.importer.xgboost.XGBoostImporter;
import com.google.common.collect.ImmutableList;
import com.yahoo.config.application.api.ApplicationPackage;
import com.yahoo.config.model.application.provider.BaseDeployLogger;
import com.yahoo.config.model.deploy.TestProperties;
import com.yahoo.io.IOUtils;
import com.yahoo.path.Path;
import com.yahoo.search.query.profile.QueryProfileRegistry;
import com.yahoo.searchdefinition.RankProfile;
import com.yahoo.searchdefinition.RankProfileRegistry;
import com.yahoo.searchdefinition.Search;
import com.yahoo.searchdefinition.SearchBuilder;
import com.yahoo.searchdefinition.derived.DerivedConfiguration;
import com.yahoo.searchdefinition.parser.ParseException;
import com.yahoo.searchdefinition.processing.RankingExpressionWithTensorFlowTestCase.StoringApplicationPackage;
import com.yahoo.vespa.model.VespaModel;
import com.yahoo.vespa.model.ml.ImportedModelTester;
import com.yahoo.yolean.Exceptions;
import org.junit.After;
import org.junit.Ignore;
import org.junit.Test;

import java.io.IOException;
import java.util.Optional;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.fail;

public class RankingExpressionWithBertTestCase {

    private final Path applicationDir = Path.fromString("src/test/integration/bert/");

    /** The model name */
    private final static String name = "bertsquad8";

    private final static String vespaExpression = "join(reduce(join(rename(Placeholder, (d0, d1), (d0, d2)), constant(" + name + "_Variable), f(a,b)(a * b)), sum, d2), constant(" + name + "_Variable_1), f(a,b)(a + b))";

    @After
    public void removeGeneratedModelFiles() {
        IOUtils.recursiveDeleteDir(applicationDir.append(ApplicationPackage.MODELS_GENERATED_DIR).toFile());
    }


    @Ignore
    @Test
    public void testGlobalBertModel() throws IOException {
        ImportedModelTester tester = new ImportedModelTester(name, applicationDir);
        VespaModel model = tester.createVespaModel();
//        tester.assertLargeConstant(name + "_Variable_1", model, Optional.of(10L));
//        tester.assertLargeConstant(name + "_Variable", model, Optional.of(7840L));

        // At this point the expression is stored - copy application to another location which do not have a models dir
        Path storedAppDir = applicationDir.append("copy");
        try {
            storedAppDir.toFile().mkdirs();
            IOUtils.copy(applicationDir.append("services.xml").toString(), storedAppDir.append("services.xml").toString());
            IOUtils.copyDirectory(applicationDir.append(ApplicationPackage.MODELS_GENERATED_DIR).toFile(),
                    storedAppDir.append(ApplicationPackage.MODELS_GENERATED_DIR).toFile());
            ImportedModelTester storedTester = new ImportedModelTester(name, storedAppDir);
            VespaModel storedModel = storedTester.createVespaModel();
//            tester.assertLargeConstant(name + "_Variable_1", storedModel, Optional.of(10L));
//            tester.assertLargeConstant(name + "_Variable", storedModel, Optional.of(7840L));
        }
        finally {
            IOUtils.recursiveDeleteDir(storedAppDir.toFile());
        }
    }

    @Ignore
    @Test
    public void testBertRankProfile() throws Exception {
        StoringApplicationPackage application = new StoringApplicationPackage((applicationDir));

        ImmutableList<MlModelImporter> importers = ImmutableList.of(new TensorFlowImporter(),
                new OnnxImporter(),
                new LightGBMImporter(),
                new XGBoostImporter());

        String rankProfiles = "  rank-profile my_profile {\n" +
                "    first-phase {\n" +
                "      expression: onnx('bertsquad8.onnx', 'default', 'unstack')" +
                "    }\n" +
                "  }";

        RankProfileRegistry rankProfileRegistry = new RankProfileRegistry();
        QueryProfileRegistry queryProfileRegistry = application.getQueryProfiles();

        SearchBuilder builder = new SearchBuilder(application, rankProfileRegistry, queryProfileRegistry);
        String sdContent = "search test {\n" +
                "  document test {\n" +
                "        field unique_ids type tensor(d0[1]) {\n" +
                "            indexing: summary | attribute\n" +
                "        }\n" +
                "        field input_ids type tensor(d0[1],d1[256]) {\n" +
                "            indexing: summary | attribute\n" +
                "        }\n" +
                "        field input_mask type tensor(d0[1],d1[256]) {\n" +
                "            indexing: summary | attribute\n" +
                "        }\n" +
                "        field segment_ids type tensor(d0[1],d1[256]) {\n" +
                "            indexing: summary | attribute\n" +
                "        }" +
                "  }\n" +
                "  rank-profile my_profile inherits default {\n" +
                "        function inline unique_ids_raw_output___9() {\n" +
                "            expression: attribute(unique_ids)\n" +
                "        }\n" +
                "        function inline input_ids() {\n" +
                "            expression: attribute(input_ids)\n" +
                "        }\n" +
                "        function inline input_mask() {\n" +
                "            expression: attribute(input_mask)\n" +
                "        }\n" +
                "        function inline segment_ids() {\n" +
                "            expression: attribute(segment_ids)\n" +
                "        }\n" +
                "        first-phase {\n" +
                "            expression: onnx(\"bertsquad8.onnx\", \"default\", \"unstack\") \n" +
                "        }\n" +
                "    }" +
                "}";
        builder.importString(sdContent);
        builder.build();
        Search search = builder.getSearch();

        RankProfile compiled = rankProfileRegistry.get(search, "my_profile")
                .compile(queryProfileRegistry,
                        new ImportedMlModels(applicationDir.toFile(), importers));

        DerivedConfiguration config = new DerivedConfiguration(search,
                new BaseDeployLogger(),
                new TestProperties(),
                rankProfileRegistry,
                queryProfileRegistry,
                new ImportedMlModels());

        config.export("/Users/lesters/temp/bert/idea/");

//        fixture.assertFirstPhaseExpression(vespaExpression, "my_profile");
        System.out.println("Joda");
    }

    private RankProfileSearchFixture fixtureWith(String placeholderExpression, String firstPhaseExpression,
                                                 String constant, String field) {
        return fixtureWith(placeholderExpression, firstPhaseExpression, constant, field, "Placeholder",
                           new StoringApplicationPackage(applicationDir));
    }

    private RankProfileSearchFixture uncompiledFixtureWith(String rankProfile, StoringApplicationPackage application) {
        try {
            return new RankProfileSearchFixture(application, application.getQueryProfiles(),
                                                rankProfile, null, null);
        }
        catch (ParseException e) {
            throw new IllegalArgumentException(e);
        }
    }

    private RankProfileSearchFixture fixtureWith(String functionExpression,
                                                 String firstPhaseExpression,
                                                 String constant,
                                                 String field,
                                                 String functionName,
                                                 StoringApplicationPackage application) {
        try {
            RankProfileSearchFixture fixture = new RankProfileSearchFixture(
                    application,
                    application.getQueryProfiles(),
                    "  rank-profile my_profile {\n" +
                            "    function " + functionName + "() {\n" +
                            "      expression: " + functionExpression +
                            "    }\n" +
                            "    first-phase {\n" +
                            "      expression: " + firstPhaseExpression +
                            "    }\n" +
                            "  }",
                    constant,
                    field);
            fixture.compileRankProfile("my_profile", applicationDir.append("models"));
            return fixture;
        }
        catch (ParseException e) {
            throw new IllegalArgumentException(e);
        }
    }

}
