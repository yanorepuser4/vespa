// Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
#include "metricsnapshot.h"
#include "metricmanager.h"
#include <cassert>

#include <vespa/log/log.h>
LOG_SETUP(".metrics.snapshot");

using vespalib::to_string;

namespace metrics {

MetricSnapshot::MetricSnapshot(const Metric::String& name)
    : _name(name),
      _period(0),
      _fromTime(default_system_time),
      _toTime(default_system_time),
      _snapshot(new MetricSet("top", {}, "", nullptr)),
      _metrics()
{
}

MetricSnapshot::MetricSnapshot(const Metric::String& name, uint32_t period, const MetricSet& source, bool copyUnset)
    : _name(name),
      _period(period),
      _fromTime(default_system_time),
      _toTime(default_system_time),
      _snapshot(),
      _metrics()
{
    Metric* m = source.clone(_metrics, Metric::INACTIVE, 0, copyUnset);
    assert(m->isMetricSet());
    _snapshot.reset(static_cast<MetricSet*>(m));
    _metrics.shrink_to_fit();
}

MetricSnapshot::~MetricSnapshot() = default;

void
MetricSnapshot::reset() {
    reset(default_system_time);
}
void
MetricSnapshot::reset(system_time currentTime)
{
    _fromTime = currentTime;
    _toTime = default_system_time;
    _snapshot->reset();
}

void
MetricSnapshot::recreateSnapshot(const MetricSet& metrics, bool copyUnset)
{
    std::vector<Metric::UP> newMetrics;
    Metric* m = metrics.clone(newMetrics, Metric::INACTIVE, 0, copyUnset);
    assert(m->isMetricSet());
    std::unique_ptr<MetricSet> newSnapshot(static_cast<MetricSet*>(m));
    newSnapshot->reset();
    _snapshot->addToSnapshot(*newSnapshot, newMetrics);
    _snapshot = std::move(newSnapshot);
    _metrics.swap(newMetrics);
    _metrics.shrink_to_fit();
}

void
MetricSnapshot::addMemoryUsage(MemoryConsumption& mc) const
{
    ++mc._snapshotCount;
    mc._snapshotName += mc.getStringMemoryUsage(_name, mc._snapshotNameUnique);
    mc._snapshotMeta += sizeof(MetricSnapshot) + _metrics.capacity() * sizeof(Metric::SP);
    _snapshot->addMemoryUsage(mc);
}

MetricSnapshotSet::MetricSnapshotSet(const Metric::String& name, uint32_t period, uint32_t count,
                                     const MetricSet& source, bool snapshotUnsetMetrics)
    : _count(count),
      _builderCount(0),
      _current(std::make_unique<MetricSnapshot>(name, period, source, snapshotUnsetMetrics)),
      _building(count == 1 ? nullptr : new MetricSnapshot(name, period, source, snapshotUnsetMetrics))
{
    _current->reset();
    if (_building.get()) _building->reset();
}

MetricSnapshot&
MetricSnapshotSet::getNextTarget()
{
    if (_count == 1) return *_current;
    return *_building;
}

bool
MetricSnapshotSet::haveCompletedNewPeriod(system_time newFromTime)
{
    if (_count == 1) {
        _current->setToTime(newFromTime);
        return true;
    }
    _building->setToTime(newFromTime);
    // If not time to roll yet, just return
    if (++_builderCount < _count) return false;
    // Building buffer done. Use that as current and reset current.
    MetricSnapshot::UP tmp(std::move(_current));
    _current = std::move(_building);
    _building = std::move(tmp);
    _building->reset(newFromTime);
    _builderCount = 0;
    return true;
}

bool
MetricSnapshotSet::timeForAnotherSnapshot(system_time currentTime) {
    system_time lastTime = getToTime();
    vespalib::duration period = vespalib::from_s(getPeriod());
    if (currentTime >= lastTime + period) {
        if (currentTime >= lastTime + 2 * period) {
            LOG(warning, "Metric snapshot set %s was asked if it was time for another snapshot, a whole period beyond "
                         "when it should have been done (Last update was at time %s, current time is %s and period "
                         "is %u). Clearing data and updating time to current time.",
                getName().c_str(), to_string(lastTime).c_str(), to_string(currentTime).c_str(), getPeriod());
            reset(currentTime);
        }
        return true;
    }
    return false;
}

void
MetricSnapshotSet::reset(system_time currentTime) {
    if (_count != 1) _building->reset(currentTime);
    _current->reset(currentTime);
    _builderCount = 0;
}

void
MetricSnapshotSet::recreateSnapshot(const MetricSet& metrics, bool copyUnset)
{
    if (_count != 1) _building->recreateSnapshot(metrics, copyUnset);
    _current->recreateSnapshot(metrics, copyUnset);
}

void
MetricSnapshotSet::addMemoryUsage(MemoryConsumption& mc) const
{
    ++mc._snapshotSetCount;
    mc._snapshotSetMeta += sizeof(MetricSnapshotSet);
    if (_count != 1) _building->addMemoryUsage(mc);
    _current->addMemoryUsage(mc);
}

void
MetricSnapshotSet::setFromTime(system_time fromTime)
{
    if (_count != 1) _building->setFromTime(fromTime);
    _current->setFromTime(fromTime);
}

} // metrics
