// Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package com.yahoo.prelude.query;

import java.nio.ByteBuffer;
import java.util.Objects;

/**
 * A word item which cannot provide its own index (field) name, but will always query the index
 * specified by the parent item it is added to.
 *
 * @author baldersheim
 */
public class PureWeightedString extends PureWeightedItem  {

    private final String value;

    public PureWeightedString(String value) {
        this(value, 100);
    }

    public PureWeightedString(String value, int weight) {
        super(weight);
        this.value = value;
    }

    @Override
    public ItemType getItemType() {
        return ItemType.PURE_WEIGHTED_STRING;
    }

    @Override
    protected void encodeThis(ByteBuffer buffer) {
        super.encodeThis(buffer);
        putString(value, buffer);
    }

    @Override
    public int getTermCount() {
        return 1;
    }

    @Override
    protected void appendBodyString(StringBuilder buffer) {
        buffer.append(value);
        super.appendBodyString(buffer);
    }

    public String getString() {
        return value;
    }

    @Override
    public boolean equals(Object other) {
        if ( ! super.equals(other)) return false;
        return Objects.equals(value, ((PureWeightedString)other).value);
    }

    @Override
    public int hashCode() {
        return Objects.hash(super.hashCode(), value);
    }

}
