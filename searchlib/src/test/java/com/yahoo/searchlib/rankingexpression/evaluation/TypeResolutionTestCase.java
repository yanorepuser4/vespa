// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

package com.yahoo.searchlib.rankingexpression.evaluation;

import com.yahoo.searchlib.rankingexpression.RankingExpression;
import com.yahoo.searchlib.rankingexpression.Reference;
import com.yahoo.searchlib.rankingexpression.parser.ParseException;
import com.yahoo.tensor.TensorType;
import com.yahoo.tensor.evaluation.TypeContext;
import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;

/**
 * @author bratseth
 */
public class TypeResolutionTestCase {

    @Test
    public void testTypeResolution() {
        MapTypeContext context = new MapTypeContext();
        context.setType(Reference.simple("query", "x1"),
                        TensorType.fromSpec("tensor(x[])"));
        context.setType(Reference.simple("query", "x2"),
                        TensorType.fromSpec("tensor(x[10])"));
        context.setType(Reference.simple("query", "y1"),
                        TensorType.fromSpec("tensor(y[])"));
        context.setType(Reference.simple("query", "xy1"),
                        TensorType.fromSpec("tensor(x[10],y[])"));
        context.setType(Reference.simple("query", "xy2"),
                        TensorType.fromSpec("tensor(x[],y[10])"));

        assertType("tensor(x[])", "query(x1)", context);
        assertType("tensor(x[])", "if (1>0, query(x1), query(x2))", context);
        assertType("tensor(x[],y[])", "if (1>0, query(xy1), query(xy2))", context);
        assertIncompatibleType("if (1>0, query(x1), query(y1))", context);
    }

    private void assertType(String type, String expression, TypeContext<Reference> context) {
        try {
            assertEquals(TensorType.fromSpec(type), new RankingExpression(expression).type(context));
        }
        catch (ParseException e) {
            throw new RuntimeException(e);
        }
    }

    private void assertIncompatibleType(String expression, TypeContext<Reference> context) {
        try {
            new RankingExpression(expression).type(context);
            fail("Expected type incompatibility exception");
        }
        catch (IllegalArgumentException expected) {
            assertEquals("An if expression must produce compatible types in both alternatives, " +
                         "but the 'true' type is tensor(x[]) while the 'false' type is tensor(y[])" +
                         "\n'true' branch: query(x1)" +
                         "\n'false' branch: query(y1)",
                         expected.getMessage());
        }
        catch (ParseException e) {
            throw new RuntimeException(e);
        }
    }

}
