import { StringOutputParser, CommaSeparatedListOutputParser, StructuredOutputParser } from '@langchain/core/output_parsers';
import { ChatPromptTemplate } from '@langchain/core/prompts';
import { RunnableSequence } from '@langchain/core/runnables';
import { ChatOpenAI } from '@langchain/openai';

const model = new ChatOpenAI({
	modelName: 'gpt-4o',
	temperature: 0.8,
	maxTokens: 700,
});

async function stringParser() {
	const prompt = ChatPromptTemplate.fromTemplate('Write a short description for the following product: {product_name}');

	const parser = new StringOutputParser();

	const chain = prompt.pipe(model).pipe(parser);

	const response = await chain.invoke({
		product_name: 'iPhone',
	});

	console.log(response);
}

async function commaSeparatedParser() {
	const prompt = ChatPromptTemplate.fromTemplate('Provide the first 5 ingredients, sperated by comma, for : {word}');

	const parser = new CommaSeparatedListOutputParser();

	const chain = RunnableSequence.from([prompt, model, parser]); // another way to chain this things

	const response = await chain.invoke({
		word: 'paubhaji',
	});

	console.log(response);
}

async function structuredParser() {
	const templatePrompt = ChatPromptTemplate.fromTemplate(
		'Extract information from the following phrase. Formatting instruction: {format_instructions}. Phrase: {phrase}'
	);

	const outputParser = StructuredOutputParser.fromNamesAndDescriptions({
		name: 'Sentiment analysis',
		description: 'Analyse the sentiment of the input text',
	});

	const chain = templatePrompt.pipe(model).pipe(outputParser);

	const response = await chain.invoke({
		phrase: 'I love programming',
		format_instructions: outputParser.getFormatInstructions(),
	});

	console.log(response);
}

stringParser();
commaSeparatedParser();
structuredParser();
