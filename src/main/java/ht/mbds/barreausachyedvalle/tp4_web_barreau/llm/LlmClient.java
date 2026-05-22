package ht.mbds.barreausachyedvalle.tp4_web_barreau.llm;

import dev.langchain4j.rag.query.transformer.QueryTransformer;
import dev.langchain4j.rag.query.transformer.CompressingQueryTransformer;
import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.RetrievalAugmentor;
import dev.langchain4j.rag.query.router.QueryRouter;
import dev.langchain4j.rag.query.router.LanguageModelQueryRouter;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentSplitter;
import dev.langchain4j.data.document.loader.ClassPathDocumentLoader;
import dev.langchain4j.data.document.parser.apache.tika.ApacheTikaDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.allminilml6v2.AllMiniLmL6V2EmbeddingModel;
import dev.langchain4j.model.output.Response;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import java.util.Map;
import java.util.List;
import dev.langchain4j.data.message.SystemMessage;
import dev.langchain4j.memory.ChatMemory;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.googleai.GoogleAiGeminiChatModel;
import dev.langchain4j.service.AiServices;
import jakarta.enterprise.context.Dependent;

import java.io.Serializable;

@Dependent
public class LlmClient implements Serializable {

    private String systemRole;
    private Assistant assistant;
    private ChatMemory chatMemory;

    public LlmClient() {
        String apiKey = System.getenv("GEMINI_KEY");

        ChatModel model = GoogleAiGeminiChatModel.builder()
                .apiKey(apiKey)
                .modelName("gemini-2.5-flash-lite")
                .build();


        this.chatMemory = MessageWindowChatMemory.withMaxMessages(20);
        ContentRetriever ragRetriever =
                createRetriever("rag.pdf");

        ContentRetriever hadoopRetriever =
                createRetriever("HadoopSparkMapReduce_1.pdf");
        QueryRouter queryRouter =
                new LanguageModelQueryRouter(
                        model,
                        Map.of(
                                ragRetriever, "Documents sur l'intelligence artificielle, le RAG et le fine-tuning",
                                hadoopRetriever, "Documents sur Hadoop, Spark, MapReduce et HDFS"
                        )
                );
        QueryTransformer queryTransformer =
                new CompressingQueryTransformer(model);
        RetrievalAugmentor retrievalAugmentor =
                DefaultRetrievalAugmentor.builder()
                        .queryRouter(queryRouter)
                        .queryTransformer(queryTransformer)
                        .build();

        this.assistant = AiServices.builder(Assistant.class)
                .chatModel(model)
                .chatMemory(chatMemory)
                .retrievalAugmentor(retrievalAugmentor)
                .build();
    }

    public void setSystemRole(String role) {
        this.systemRole = role;
        this.chatMemory.clear();
        this.chatMemory.add(SystemMessage.from(role));
    }

    public String envoyerQuestion(String role, String question) {
        if (role != null) {
            setSystemRole(role);
        }

        return assistant.chat(question);
    }
    private ContentRetriever createRetriever(String pdfName) {

        Document document = ClassPathDocumentLoader.loadDocument(
                pdfName,
                new ApacheTikaDocumentParser()
        );

        DocumentSplitter splitter =
                DocumentSplitters.recursive(300, 30);

        List<TextSegment> segments = splitter.split(document);

        EmbeddingModel embeddingModel =
                new AllMiniLmL6V2EmbeddingModel();

        Response<List<Embedding>> response =
                embeddingModel.embedAll(segments);

        List<Embedding> embeddings = response.content();

        EmbeddingStore<TextSegment> embeddingStore =
                new InMemoryEmbeddingStore<>();

        embeddingStore.addAll(embeddings, segments);

        return EmbeddingStoreContentRetriever.builder()
                .embeddingStore(embeddingStore)
                .embeddingModel(embeddingModel)
                .maxResults(2)
                .minScore(0.5)
                .build();
    }

}