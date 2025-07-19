#include "prob_gar_chase.h"
#include "../types.h"
#include <gflags/gflags.h>
#include <gflags/gflags_declare.h>
#include <glog/logging.h>

int main(int argc, char* argv[]) {
  FLAGS_stderrthreshold = 0;

  gflags::SetUsageMessage(
      "Usage: mpiexec [mpi_opts] ./prob_gar_chase [grape_opts]");
  if (argc == 1) {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "prob_gar_chase");
    exit(1);
  }
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  gflags::ShutDownCommandLineFlags();

  google::InitGoogleLogging("prob_gar_chase");
  google::InstallFailureSignalHandler();

  grape::Init();

  grape::Run<int64_t, uint32_t, grape::Data, grape::EdgeData>();

  grape::Finalize();

  google::ShutdownGoogleLogging();
}
