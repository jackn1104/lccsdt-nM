#pragma once
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>
#if defined(__linux__)
#include <fcntl.h>
#include <unistd.h>
#endif

namespace localized_cc::detail {
inline std::uint64_t sector_checksum(const std::string& bytes) {
  std::uint64_t hash=14695981039346656037ULL;
  for(unsigned char c:bytes) { hash^=c; hash*=1099511628211ULL; }
  return hash;
}
inline std::vector<double> read_sector_checkpoint(const std::string& path,
    const std::string& identity, std::size_t count) {
  if(!std::filesystem::exists(path)) return std::vector<double>(count,0.);
  if(std::filesystem::file_size(path)>1024*1024)
    throw std::runtime_error("triples sector checkpoint too large");
  std::ifstream in(path,std::ios::binary);
  if(!in) throw std::runtime_error("cannot open triples sector checkpoint");
  std::string bytes((std::istreambuf_iterator<char>(in)),{});
  const std::string prefix="LCCSDT-SECTOR-1\n"+identity+"\n";
  const auto payload=count*sizeof(double);
  if(!bytes.starts_with(prefix)||bytes.size()!=prefix.size()+payload+sizeof(std::uint64_t))
    throw std::runtime_error("triples sector checkpoint identity/size mismatch");
  std::uint64_t stored{};
  std::memcpy(&stored,bytes.data()+bytes.size()-sizeof(stored),sizeof(stored));
  if(stored!=sector_checksum(bytes.substr(0,bytes.size()-sizeof(stored))))
    throw std::runtime_error("triples sector checkpoint checksum mismatch");
  std::vector<double> values(count);
  std::memcpy(values.data(),bytes.data()+prefix.size(),payload);
  for(auto x:values) if(!std::isfinite(x)) throw std::runtime_error("nonfinite triples sector checkpoint");
  return values;
}
inline void write_sector_checkpoint(const std::string& path,const std::string& identity,
    const std::vector<double>& values) {
#if defined(__linux__)
  for(auto x:values) if(!std::isfinite(x)) throw std::runtime_error("nonfinite triples sector checkpoint");
  std::string bytes="LCCSDT-SECTOR-1\n"+identity+"\n";
  bytes.append(reinterpret_cast<const char*>(values.data()),values.size()*sizeof(double));
  const auto hash=sector_checksum(bytes);
  bytes.append(reinterpret_cast<const char*>(&hash),sizeof(hash));
  auto temp=path+".tmp.XXXXXX";
  int fd=::mkstemp(temp.data());
  if(fd<0) throw std::runtime_error("cannot create triples checkpoint temporary file");
  try {
    std::size_t offset=0;
    while(offset<bytes.size()) {
      const auto n=::write(fd,bytes.data()+offset,bytes.size()-offset);
      if(n<=0) throw std::runtime_error("triples checkpoint write failed");
      offset+=static_cast<std::size_t>(n);
    }
    if(::fsync(fd)!=0) throw std::runtime_error("triples checkpoint fsync failed");
    const int close_status=::close(fd); fd=-1;
    if(close_status!=0) throw std::runtime_error("triples checkpoint close failed");
    std::filesystem::rename(temp,path);
    auto parent=std::filesystem::path(path).parent_path();
    if(parent.empty()) parent=".";
    const int directory=::open(parent.c_str(),O_RDONLY|O_DIRECTORY);
    if(directory<0) throw std::runtime_error("triples checkpoint directory open failed");
    const int sync_status=::fsync(directory); ::close(directory);
    if(sync_status!=0) throw std::runtime_error("triples checkpoint directory fsync failed");
  } catch(...) {
    if(fd>=0) ::close(fd);
    ::unlink(temp.c_str());
    throw;
  }
#else
  (void)path; (void)identity; (void)values;
  throw std::runtime_error("persistent triples sector checkpoints require Linux");
#endif
}
}
