import 'package:infergo/infergo.dart';

void main(List<String> args) {
  final llm = LLM(args.isNotEmpty ? args[0] : 'model.gguf');
  print(llm.generate('What is Dart?', maxTokens: 64));
  llm.close();
}
