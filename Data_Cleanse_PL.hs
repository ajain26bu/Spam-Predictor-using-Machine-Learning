import Data.Char (isSpace)
import Data.Char (toLower)
trim :: String -> String
trim = f . f
  where f = reverse . dropWhile isSpace
lowercase :: String -> String
lowercase = map toLower
main :: IO ()
main = do
  input <- readFile "a.txt"
  let punctuations = [ '!', '"', '#', '$', '%', '(', ')', '.', ',', '?', ':', ';']
  let removePunctuation = filter (`notElem` punctuations)   
  let specialSymbols = ['/', '-', '*','&']
  let replaceSpecialSymbols = map $ do (\c ->if c `elem` specialSymbols then ' ' else c)        
  let output = (removePunctuation.replaceSpecialSymbols.trim.lowercase) input
  writeFile "out.csv" output