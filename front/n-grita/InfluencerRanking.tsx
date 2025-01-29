import type React from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"
import { influencers } from "./data"
import type { Influencer } from "./types"
import { Star, Trophy } from "lucide-react"

const InfluencerRanking: React.FC = () => {
  const sortedInfluencers = [...influencers].sort((a, b) => b.rate - a.rate)
  const topThree = sortedInfluencers.slice(0, 3)
  const others = sortedInfluencers.slice(3)

  const renderStars = (rating: number) => {
    return Array(5)
      .fill(0)
      .map((_, i) => (
        <Star
          key={i}
          className={`w-4 h-4 ${i < Math.round(rating) ? "text-yellow-400 fill-yellow-400" : "text-gray-300"}`}
        />
      ))
  }

  const PodiumCard: React.FC<{ influencer: Influencer; position: number }> = ({ influencer, position }) => (
    <Card className={`w-full max-w-sm mx-auto ${position === 2 ? "mt-8" : position === 3 ? "mt-16" : ""}`}>
      <CardHeader>
        <CardTitle className="flex justify-between items-center">
          <span className="flex items-center">
            <Trophy
              className={`w-6 h-6 mr-2 ${position === 1 ? "text-yellow-400" : position === 2 ? "text-gray-400" : "text-orange-400"}`}
            />
            {position}º Lugar
          </span>
          <span className="text-sm text-muted-foreground">@{influencer.nickname}</span>
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="flex flex-col items-center">
          <Avatar className="w-20 h-20 mb-4">
            <AvatarImage src={`https://i.pravatar.cc/150?u=${influencer.id}`} />
            <AvatarFallback>{influencer.name.charAt(0)}</AvatarFallback>
          </Avatar>
          <h3 className="text-lg font-semibold mb-2">{influencer.name}</h3>
          <div className="flex items-center mb-2">
            {renderStars(influencer.rate)}
            <span className="ml-2">{influencer.rate.toFixed(1)}</span>
          </div>
          <p className="text-sm text-muted-foreground text-center">{influencer.advice_description}</p>
        </div>
      </CardContent>
    </Card>
  )

  return (
    <div className="container mx-auto p-4">
      <h1 className="text-3xl font-bold mb-8 text-center">Ranking de Influenciadores</h1>

      <div className="mb-12">
        <h2 className="text-2xl font-semibold mb-6 text-center">Pódio dos Top 3</h2>
        <div className="flex flex-col md:flex-row justify-center items-end space-y-4 md:space-y-0 md:space-x-4">
          <PodiumCard influencer={topThree[1]} position={2} />
          <PodiumCard influencer={topThree[0]} position={1} />
          <PodiumCard influencer={topThree[2]} position={3} />
        </div>
      </div>

      <div>
        <h2 className="text-2xl font-semibold mb-6 text-center">Outros Influenciadores</h2>
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Posição</TableHead>
              <TableHead>Nome</TableHead>
              <TableHead>Nickname</TableHead>
              <TableHead>Rating</TableHead>
              <TableHead>Descrição</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {others.map((influencer, index) => (
              <TableRow key={influencer.id}>
                <TableCell>{index + 4}</TableCell>
                <TableCell className="font-medium">{influencer.name}</TableCell>
                <TableCell>@{influencer.nickname}</TableCell>
                <TableCell>
                  <div className="flex items-center">
                    {renderStars(influencer.rate)}
                    <span className="ml-2">{influencer.rate.toFixed(1)}</span>
                  </div>
                </TableCell>
                <TableCell className="max-w-xs truncate">{influencer.work_description}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </div>
    </div>
  )
}

export default InfluencerRanking

