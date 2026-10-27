#ifndef NOCDAS_MESSAGE_KIND_HPP_
#define NOCDAS_MESSAGE_KIND_HPP_

// Wire/application meaning is independent of endpoint queue and virtual network.
enum class MessageKind : int {
    DataRequest = 0, DataResponse = 1, Result = 2, Ack = 3,
    Distribution = 4, Compute = 5, TerminalOperands = 7
};
enum class ReceiveClass : int { RequestsAndData = 0, ResultsAndAcks = 1 };
enum class VirtualNetwork : int { Memory = 0, Compute = 1 };
constexpr int queueIndex(ReceiveClass value) { return static_cast<int>(value); }
constexpr int vnIndex(VirtualNetwork value) { return static_cast<int>(value); }
#endif
