#pragma once

#include "quant/execution/execution_gateway.hpp"

#include <string_view>

struct SessionSummary
{
    double initial_capital{0.0};
    double start_price{0.0};
    double final_price{0.0};
    double sharpe_ratio{0.0};
};

class ISessionObserver
{
public:
    virtual ~ISessionObserver() = default;
    virtual void on_session_start() {}
    virtual void on_close_fill(std::string_view /*label*/, const ExecutionFill& /*fill*/) {}
    virtual void on_execution_fill(const ExecutionFill& /*fill*/) {}
    virtual void on_session_end(const SessionSummary& /*summary*/, const AccountState&) {}
};

class NullSessionObserver final : public ISessionObserver
{
};
