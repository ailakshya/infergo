Gem::Specification.new do |s|
  s.name        = 'infergo'
  s.version     = '1.0.0'
  s.summary     = 'infergo — native AI inference for Ruby'
  s.description = 'Direct FFI bindings to libinfer_api.so. LLM, embedding, vector search, BM25.'
  s.authors     = ['infergo']
  s.license     = 'Apache-2.0'
  s.files       = ['lib/infergo.rb']
  s.add_dependency 'ffi', '~> 1.15'
  s.required_ruby_version = '>= 2.7'
end
