#include "gar_overlap.h"
#include "../types.h"
#include <gflags/gflags.h>
#include <gflags/gflags_declare.h>
#include <glog/logging.h>

int main(int argc, char* argv[]) {
  FLAGS_stderrthreshold = 0;

  gflags::SetUsageMessage(
      "Usage: mpiexec [mpi_opts] ./gar_overlap [grape_opts]");
  if (argc == 1) {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "gar_overlap");
    exit(1);
  }
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  gflags::ShutDownCommandLineFlags();

  google::InitGoogleLogging("gar_overlap");
  google::InstallFailureSignalHandler();

  grape::Init();

  grape::Run<int64_t, uint32_t, grape::Data, grape::EdgeData>();

  grape::Finalize();

  google::ShutdownGoogleLogging();
}
