// Copyright 2019 Oath Inc. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package com.yahoo.vespa.hosted.node.admin.task.util.env;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.TreeSet;

/**
 * Rewrites default-env.txt files.
 *
 * @author bjorncs
 */
public class DefaultEnvRewriter {

    private final Map<String, Operation> operations = new TreeMap<>();
    private final Path defaultEnvFile;

    public DefaultEnvRewriter(Path defaultEnvFile) {
        this.defaultEnvFile = defaultEnvFile;
    }

    public DefaultEnvRewriter addOverride(String name, String value) {
        return addOperation("override", name, value);
    }

    public DefaultEnvRewriter addFallback(String name, String value) {
        return addOperation("fallback", name, value);
    }

    public DefaultEnvRewriter addUnset(String name) {
        return addOperation("unset", name, null);
    }

    private DefaultEnvRewriter addOperation(String action, String name, String value) {
        if (operations.containsKey(name)) {
            throw new IllegalArgumentException(String.format("Operation on variable '%s' already added", name));
        }
        operations.put(name, new Operation(action, name, value));
        return this;
    }

    public boolean converge() throws IOException {
        List<String> defaultEnvLines = Files.readAllLines(defaultEnvFile);
        List<String> newDefaultEnvLines = new ArrayList<>();
        Set<String> seenNames = new TreeSet<>();
        for (String line : defaultEnvLines) {
            String[] items = line.split(" ");
            if (items.length < 2) {
                throw new IllegalArgumentException(String.format("Invalid line in file '%s': %s", defaultEnvFile, line));
            }
            String name = items[1];
            if (!seenNames.contains(name)) { // implicitly removes duplicated variables
                seenNames.add(name);
                Operation operation = operations.get(name);
                if (operation != null) {
                    newDefaultEnvLines.add(operation.toLine());
                } else {
                    newDefaultEnvLines.add(line);
                }
            }
        }
        for (var operation : operations.values()) {
            if (!seenNames.contains(operation.name)) {
                newDefaultEnvLines.add(operation.toLine());
            }
        }
        if (defaultEnvLines.equals(newDefaultEnvLines)) {
            return false;
        } else {
            Files.write(defaultEnvFile, newDefaultEnvLines);
            return true;
        }
    }

    private static class Operation {
        final String action;
        final String name;
        final String value;

        Operation(String action, String name, String value) {
            this.action = action;
            this.name = name;
            this.value = value;
        }

        String toLine() {
            if (action.equals("unset")) {
                return "unset " + name;
            }
            return action + " " + name + " " + value;
        }
    }
}


