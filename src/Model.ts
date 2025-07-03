import { ChatOpenAI } from '@langchain/openai';

const model = new ChatOpenAI({
	modelName: 'gpt-4o',
	temperature: 0.8,
	maxTokens: 700,
	// verbose: true,
});

async function main() {
	const res1 = await model.invoke('What is the capital of France?');
	console.log(res1.content);

	//Batch method request happen in parellel so all this question at the sametime to LLM
	const res2 = await model.batch(['What is the capital of Luxembourg?', 'What is the capital of Germany?', 'What is the capital of Italy?']);
	for (const res of res2) {
		console.log(res.content);
	}

	// Stream as the name suggested you will data is output like someone typing
	const res3 = await model.stream('Give me 4 good books related to human psychology?');
	for await (const chunk of res3) {
		console.log(chunk.content);
	}
}

main();
