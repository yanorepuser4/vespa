// Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package com.yahoo.vespa.clustercontroller.core;

import com.yahoo.jrt.Request;
import com.yahoo.jrt.Spec;
import com.yahoo.jrt.StringValue;
import com.yahoo.jrt.Supervisor;
import com.yahoo.jrt.Target;
import com.yahoo.jrt.Transport;
import com.yahoo.jrt.slobrok.server.Slobrok;
import com.yahoo.log.LogSetup;
import com.yahoo.vdslib.distribution.ConfiguredNode;
import com.yahoo.vdslib.state.ClusterState;
import com.yahoo.vdslib.state.Node;
import com.yahoo.vdslib.state.NodeState;
import com.yahoo.vdslib.state.NodeType;
import com.yahoo.vdslib.state.State;
import com.yahoo.vespa.clustercontroller.core.database.DatabaseHandler;
import com.yahoo.vespa.clustercontroller.core.database.ZooKeeperDatabaseFactory;
import com.yahoo.vespa.clustercontroller.core.rpc.RPCCommunicator;
import com.yahoo.vespa.clustercontroller.core.rpc.RpcServer;
import com.yahoo.vespa.clustercontroller.core.rpc.SlobrokClient;
import com.yahoo.vespa.clustercontroller.core.status.StatusHandler;
import com.yahoo.vespa.clustercontroller.core.testutils.WaitCondition;
import com.yahoo.vespa.clustercontroller.core.testutils.WaitTask;
import com.yahoo.vespa.clustercontroller.core.testutils.Waiter;
import com.yahoo.vespa.clustercontroller.utils.util.NoMetricReporter;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.api.extension.ExtensionContext;
import org.junit.jupiter.api.extension.TestWatcher;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.TreeSet;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static org.junit.jupiter.api.Assertions.fail;

/**
 * @author Håkon Humberset
 */
@ExtendWith(FleetControllerTest.CleanupZookeeperLogsOnSuccess.class)
public abstract class FleetControllerTest implements Waiter {

    private static final Logger log = Logger.getLogger(FleetControllerTest.class.getName());
    private static final int DEFAULT_NODE_COUNT = 10;

    private final Duration timeout = Duration.ofSeconds(30);
    protected final FakeTimer timer = new FakeTimer();

    Supervisor supervisor;
    protected Slobrok slobrok;
    protected FleetControllerOptions options;
    ZooKeeperTestServer zooKeeperServer;
    protected FleetController fleetController;
    protected List<DummyVdsNode> nodes = new ArrayList<>();
    private String testName;

    private final Waiter waiter = new Waiter.Impl(new DataRetriever() {
        @Override
        public Object getMonitor() { return timer; }
        @Override
        public FleetController getFleetController() { return fleetController; }
        @Override
        public List<DummyVdsNode> getDummyNodes() { return nodes; }
        @Override
        public Duration getTimeout() { return timeout; }
    });

    static {
        LogSetup.initVespaLogging("fleetcontroller");
    }

    public static class CleanupZookeeperLogsOnSuccess implements TestWatcher {

        public CleanupZookeeperLogsOnSuccess() {}

        @Override
        public void testFailed(ExtensionContext context, Throwable cause) {
            System.err.println("TEST FAILED - NOT cleaning up zookeeper directory");
            shutdownZooKeeper(context, false);
        }

        @Override
        public void testSuccessful(ExtensionContext context) {
            System.err.println("TEST SUCCEEDED - cleaning up zookeeper directory");
            shutdownZooKeeper(context, true);
        }

        private void shutdownZooKeeper(ExtensionContext ctx, boolean cleanupZooKeeperDir) {
            FleetControllerTest test = (FleetControllerTest) ctx.getTestInstance().orElseThrow();
            if (test.zooKeeperServer != null) {
                test.zooKeeperServer.shutdown(cleanupZooKeeperDir);
                test.zooKeeperServer = null;
            }
        }
    }

    protected void startingTest(String name) {
        System.err.println("STARTING TEST: " + name);
        testName = name;
    }

    static protected FleetControllerOptions defaultOptions(String clusterName) {
        return defaultOptions(clusterName, DEFAULT_NODE_COUNT);
    }

    static protected FleetControllerOptions defaultOptions(String clusterName, int nodeCount) {
        return defaultOptions(clusterName, IntStream.range(0, nodeCount)
                                                    .mapToObj(i -> new ConfiguredNode(i, false))
                                                    .collect(Collectors.toSet()));
    }

    static protected FleetControllerOptions defaultOptions(String clusterName, Collection<ConfiguredNode> nodes) {
        var opts = new FleetControllerOptions(clusterName, nodes);
        opts.enableTwoPhaseClusterStateActivation = true; // Enable by default, tests can explicitly disable.
        return opts;
    }

    void setUpSystem(FleetControllerOptions options) throws Exception {
        log.log(Level.FINE, "Setting up system");
        slobrok = new Slobrok();
        this.options = options;
        if (options.zooKeeperServerAddress != null) {
            zooKeeperServer = new ZooKeeperTestServer();
            this.options.zooKeeperServerAddress = zooKeeperServer.getAddress();
            log.log(Level.FINE, "Set up new zookeeper server at " + this.options.zooKeeperServerAddress);
        }
        this.options.slobrokConnectionSpecs = getSlobrokConnectionSpecs(slobrok);
    }

    FleetController createFleetController(boolean useFakeTimer, FleetControllerOptions options) throws Exception {
        var context = new TestFleetControllerContext(options);
        Timer timer = useFakeTimer ? this.timer : new RealTimer();
        var metricUpdater = new MetricUpdater(new NoMetricReporter(), options.fleetControllerIndex, options.clusterName);
        var log = new EventLog(timer, metricUpdater);
        var cluster = new ContentCluster(options.clusterName, options.nodes, options.storageDistribution);
        var stateGatherer = new NodeStateGatherer(timer, timer, log);
        var communicator = new RPCCommunicator(
                RPCCommunicator.createRealSupervisor(),
                timer,
                options.fleetControllerIndex,
                options.nodeStateRequestTimeoutMS,
                options.nodeStateRequestTimeoutEarliestPercentage,
                options.nodeStateRequestTimeoutLatestPercentage,
                options.nodeStateRequestRoundTripTimeMaxSeconds);
        var lookUp = new SlobrokClient(context, timer);
        lookUp.setSlobrokConnectionSpecs(new String[0]);
        var rpcServer = new RpcServer(timer, timer, options.clusterName, options.fleetControllerIndex, options.slobrokBackOffPolicy);
        var database = new DatabaseHandler(context, new ZooKeeperDatabaseFactory(context), timer, options.zooKeeperServerAddress, timer);

        // Setting this <1000 ms causes ECONNREFUSED on socket trying to connect to ZK server, in ZooKeeper,
        // after creating a new ZooKeeper (session).  This causes ~10s extra time to connect after connection loss.
        // Reasons unknown.  Larger values like the default 10_000 causes that much additional running time for some tests.
        database.setMinimumWaitBetweenFailedConnectionAttempts(2_000);

        var stateGenerator = new StateChangeHandler(context, timer, log);
        var stateBroadcaster = new SystemStateBroadcaster(context, timer, timer);
        var masterElectionHandler = new MasterElectionHandler(context, options.fleetControllerIndex, options.fleetControllerCount, timer, timer);

        var status = new StatusHandler.ContainerStatusPageServer();
        var controller = new FleetController(context, timer, log, cluster, stateGatherer, communicator, status, rpcServer, lookUp,
                                             database, stateGenerator, stateBroadcaster, masterElectionHandler, metricUpdater, options);
        controller.start();
        return controller;
    }

    protected void setUpFleetController(boolean useFakeTimer, FleetControllerOptions options) throws Exception {
        if (slobrok == null) setUpSystem(options);
        if (fleetController == null) {
            fleetController = createFleetController(useFakeTimer, options);
        } else {
            throw new Exception("called setUpFleetcontroller but it was already setup");
        }
    }

    void stopFleetController() throws Exception {
        if (fleetController != null) {
            fleetController.shutdown();
            fleetController = null;
        }
    }

    void startFleetController(boolean useFakeTimer) throws Exception {
        if (fleetController == null) {
            fleetController = createFleetController(useFakeTimer, options);
        } else {
            log.log(Level.WARNING, "already started fleetcontroller, not starting another");
        }
    }

    protected void setUpVdsNodes(boolean useFakeTimer, DummyVdsNodeOptions options) throws Exception {
        setUpVdsNodes(useFakeTimer, options, false);
    }
    protected void setUpVdsNodes(boolean useFakeTimer, DummyVdsNodeOptions options, boolean startDisconnected) throws Exception {
        setUpVdsNodes(useFakeTimer, options, startDisconnected, DEFAULT_NODE_COUNT);
    }
    protected void setUpVdsNodes(boolean useFakeTimer, DummyVdsNodeOptions options, boolean startDisconnected, int nodeCount) throws Exception {
        TreeSet<Integer> nodeIndexes = new TreeSet<>();
        for (int i = 0; i < nodeCount; ++i)
            nodeIndexes.add(this.nodes.size()/2 + i); // divide by 2 because there are 2 nodes (storage and distributor) per index
        setUpVdsNodes(useFakeTimer, options, startDisconnected, nodeIndexes);
    }
    protected void setUpVdsNodes(boolean useFakeTimer, DummyVdsNodeOptions options, boolean startDisconnected, Set<Integer> nodeIndexes) throws Exception {
        String[] connectionSpecs = getSlobrokConnectionSpecs(slobrok);
        for (int nodeIndex : nodeIndexes) {
            nodes.add(new DummyVdsNode(useFakeTimer ? timer : new RealTimer(), options, connectionSpecs, this.options.clusterName, true, nodeIndex));
            if ( ! startDisconnected) nodes.get(nodes.size() - 1).connect();
            nodes.add(new DummyVdsNode(useFakeTimer ? timer : new RealTimer(), options, connectionSpecs, this.options.clusterName, false, nodeIndex));
            if ( ! startDisconnected) nodes.get(nodes.size() - 1).connect();
        }
    }
    // TODO: Replace all usages of the above setUp methods with this one, and remove the nodes field

    /**
     * Creates dummy vds nodes for the list of configured nodes and returns them.
     * As two dummy nodes are created for each configured node - one distributor and one storage node -
     * the returned list is twice as large as configuredNodes.
     */
    protected List<DummyVdsNode> setUpVdsNodes(boolean useFakeTimer, DummyVdsNodeOptions options, boolean startDisconnected, List<ConfiguredNode> configuredNodes) throws Exception {
        String[] connectionSpecs = getSlobrokConnectionSpecs(slobrok);
        nodes = new ArrayList<>();
        final boolean distributor = true;
        for (ConfiguredNode configuredNode : configuredNodes) {
            nodes.add(new DummyVdsNode(useFakeTimer ? timer : new RealTimer(), options, connectionSpecs, this.options.clusterName, distributor, configuredNode.index()));
            if ( ! startDisconnected) nodes.get(nodes.size() - 1).connect();
            nodes.add(new DummyVdsNode(useFakeTimer ? timer : new RealTimer(), options, connectionSpecs, this.options.clusterName, !distributor, configuredNode.index()));
            if ( ! startDisconnected) nodes.get(nodes.size() - 1).connect();
        }
        return nodes;
    }

    static Set<Integer> asIntSet(Integer... idx) {
        return new HashSet<>(Arrays.asList(idx));
    }

    static Set<ConfiguredNode> asConfiguredNodes(Set<Integer> indices) {
        return indices.stream().map(idx -> new ConfiguredNode(idx, false)).collect(Collectors.toSet());
    }

    void waitForStateExcludingNodeSubset(String expectedState, Set<Integer> excludedNodes) throws Exception {
        // Due to the implementation details of the test base, this.waitForState() will always
        // wait until all nodes added in the test have received the latest cluster state. Since we
        // want to entirely ignore node #6, it won't get a cluster state at all and the test will
        // fail unless otherwise handled. We thus use a custom waiter which filters out nodes with
        // the sneaky index (storage and distributors with same index are treated as different nodes
        // in this context).
        Waiter subsetWaiter = new Waiter.Impl(new DataRetriever() {
            @Override
            public Object getMonitor() { return timer; }
            @Override
            public FleetController getFleetController() { return fleetController; }
            @Override
            public List<DummyVdsNode> getDummyNodes() {
                return nodes.stream()
                        .filter(n -> !excludedNodes.contains(n.getNode().getIndex()))
                        .collect(Collectors.toList());
            }
            @Override
            public Duration getTimeout() { return timeout; }
        });
        subsetWaiter.waitForState(expectedState);
    }

    static Map<NodeType, Integer> transitionTimes(int milliseconds) {
        Map<NodeType, Integer> maxTransitionTime = new TreeMap<>();
        maxTransitionTime.put(NodeType.DISTRIBUTOR, milliseconds);
        maxTransitionTime.put(NodeType.STORAGE, milliseconds);
        return maxTransitionTime;
    }

    protected void tearDownSystem() throws Exception {
        if (testName != null) {
            //log.log(Level.INFO, "STOPPING TEST " + testName);
            System.err.println("STOPPING TEST " + testName);
            testName = null;
        }
        if (supervisor != null) {
            supervisor.transport().shutdown().join();
        }
        if (fleetController != null) {
            fleetController.shutdown();
            fleetController = null;
        }
        if (nodes != null) for (DummyVdsNode node : nodes) {
            node.shutdown();
            nodes = null;
        }
        if (slobrok != null) {
            slobrok.stop();
            slobrok = null;
        }
    }

    @AfterEach
    public void tearDown() throws Exception {
        tearDownSystem();
    }

    public ClusterState waitForStableSystem() throws Exception { return waiter.waitForStableSystem(); }
    public ClusterState waitForStableSystem(int nodeCount) throws Exception { return waiter.waitForStableSystem(nodeCount); }
    public ClusterState waitForState(String state) throws Exception { return waiter.waitForState(state); }
    public ClusterState waitForStateInAllSpaces(String state) throws Exception { return waiter.waitForStateInAllSpaces(state); }
    public ClusterState waitForStateInSpace(String space, String state) throws Exception { return waiter.waitForStateInSpace(space, state); }
    public ClusterState waitForState(String state, Duration timeout) throws Exception { return waiter.waitForState(state, timeout); }
    public ClusterState waitForInitProgressPassed(Node n, double progress) { return waiter.waitForInitProgressPassed(n, progress); }
    public ClusterState waitForClusterStateIncludingNodesWithMinUsedBits(int bitcount, int nodecount) { return waiter.waitForClusterStateIncludingNodesWithMinUsedBits(bitcount, nodecount); }

    public void wait(WaitCondition condition, WaitTask task, Duration timeout) {
        waiter.wait(condition, task, timeout);
    }

    void waitForCompleteCycle() {
        fleetController.waitForCompleteCycle(timeout);
    }

    public static Set<ConfiguredNode> toNodes(Integer ... indexes) {
        return Arrays.stream(indexes)
                .map(i -> new ConfiguredNode(i, false))
                .collect(Collectors.toSet());
    }

    void setWantedState(DummyVdsNode node, State state, String reason) {
        if (supervisor == null) {
            supervisor = new Supervisor(new Transport());
        }
        NodeState ns = new NodeState(node.getType(), state);
        if (reason != null) ns.setDescription(reason);
        Target connection = supervisor.connect(new Spec("localhost", fleetController.getRpcPort()));
        Request req = new Request("setNodeState");
        req.parameters().add(new StringValue(node.getSlobrokName()));
        req.parameters().add(new StringValue(ns.serialize()));
        connection.invokeSync(req, timeout());
        if (req.isError()) {
            fail("Failed to invoke setNodeState(): " + req.errorCode() + ": " + req.errorMessage());
        }
        if (!req.checkReturnTypes("s")) {
            fail("Failed to invoke setNodeState(): Invalid return types.");
        }
    }

    static String[] getSlobrokConnectionSpecs(Slobrok slobrok) {
        String[] connectionSpecs = new String[1];
        connectionSpecs[0] = "tcp/localhost:" + slobrok.port();
        return connectionSpecs;
    }

    Duration timeout() { return timeout; }

}
