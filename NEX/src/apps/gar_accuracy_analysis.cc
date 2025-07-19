#include "gar_accuracy_analysis.h"
#include "../types.h"
#include <gflags/gflags.h>
#include <gflags/gflags_declare.h>
#include <glog/logging.h>

int main(int argc, char* argv[]) {
  FLAGS_stderrthreshold = 0;

  gflags::SetUsageMessage(
      "Usage: mpiexec [mpi_opts] ./gar_accuracy_analysis [grape_opts]");
  if (argc == 1) {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "gar_accuracy_analysis");
    exit(1);
  }
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  gflags::ShutDownCommandLineFlags();

  google::InitGoogleLogging("gar_accuracy_analysis");
  google::InstallFailureSignalHandler();

  grape::Init();

  grape::Run<int64_t, uint32_t, grape::Data, grape::EdgeData>();

  grape::Finalize();

  google::ShutdownGoogleLogging();
}
